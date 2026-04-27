"""TRL adapter: bridge ``train.env`` into ``trl.GRPOTrainer`` agent mode.

`trl.GRPOTrainer` (>= the multi-turn / agent release) does its own rollout
loop now. It calls ``model.generate`` (vLLM-backed under Unsloth), inspects
the assistant turn for ``tool_calls`` against the ``tools=[...]`` you pass
to the trainer, executes the matching Python function, appends a
``{"role": "tool", "content": <return value>}`` turn, and continues until
the model emits an answer with no tool calls (or hits
``max_completion_length``). For us this means we DO NOT call ``SqlEnv``
during training -- the trainer owns the rollout. Our env stays useful for
offline eval, smoke tests, and transcript replay; this module is the
strictly-glue layer that maps "trainer-shaped" inputs/outputs to
"env-shaped" ones.

Three pieces ship here:

1.  :func:`make_run_sql_tool` -- a closure-bound ``run_sql(sql_command)``
    that the trainer passes to the model as a tool. It returns a model-
    facing payload (capped row preview, error message, error class).
    Designed so the model sees just enough to debug its query without
    blowing the context with 5k-row scans.

2.  :func:`reconstruct_turns` -- walks a TRL completion (a list of
    role/content/tool_calls dicts) and rebuilds our :class:`Turn` shape.
    Crucially, it RE-EXECUTES each tool-call's ``sql_command`` through the
    runtime to get the full untruncated rows for scoring. This sidesteps
    "did the tool return everything?" entirely and keeps reward functions
    deterministic w.r.t. the SQL strings the model actually emitted.

3.  :func:`make_reward_funcs` -- builds one TRL-style ``reward_func`` per
    field of :class:`~train.env.rewards.RewardBreakdown`. GRPOTrainer logs
    each as its own metric, so the breakdown shows up as separate W&B /
    trackio panels at no extra wiring cost.

Plus :func:`tasks_to_dataset`, the small dataset builder that turns a
list of :class:`~train.env.tasks.SqlTask` into the columnar shape
GRPOTrainer expects (``prompt`` + arbitrary kwargs forwarded to reward
functions).

Multi-dialect note:

TRL doesn't currently thread per-row dataset context (e.g. "what
dialect is this row?") into tool functions. To still support
multi-dialect curricula in one training run, we let the *model* carry
the dialect across via an extra ``dialect`` argument on the tool. The
multi-dialect ``run_sql(sql_command, dialect)`` factory dispatches to
``runtimes[dialect]`` at call time; the tool-aware system prompt
(``trl_agent_system_prompt(..., with_dialect_arg=True)``) tells the
model to copy the prompt's "Dialect: X" tag into every call. Rewards
re-execute against the **row's** true dialect (looked up via the
dataset's ``dialect`` column), so a model that emits the wrong
``dialect=`` argument and gets misleading tool feedback is still scored
against ground truth -- the right learning signal.

Other constraints / non-goals (v0):

*   **No per-episode hard turn budget.** TRL caps on
    ``max_completion_length`` (token budget). The reward function
    truncates the transcript at the first matching turn so the
    correctness signal lines up with our offline ``SqlEnv`` semantics;
    excess tool calls beyond the matching turn are ignored, not
    rewarded, and not penalized.
*   **No ``LLMPolicy`` / ``run_episode`` at training time.** Those are for
    offline eval (e.g. against an OpenRouter endpoint). At training time
    the trainer's vLLM does the generation.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

# Tag used when training without the TRL tools API (single-turn mode).
SQL_TAG_START = "<SQL>"
SQL_TAG_END = "</SQL>"
_SQL_TAG_RE = re.compile(r"<SQL>(.*?)</SQL>", re.DOTALL | re.IGNORECASE)

from eval.validator import compare_results
from train.env.engine import DialectRuntime
from train.env.rewards import RewardBreakdown, RewardConfig, compute_reward
from train.env.types import Turn

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

# Maximum rows surfaced back to the model. Picked high enough to handle the
# golden corpus's chunkier results (top-10 group-bys, the occasional 50-row
# join) but low enough that a 5k-row table scan doesn't blow the context.
# The reward function re-executes SQL to get the full rows; this cap is
# for the model's eyes only.
DEFAULT_PREVIEW_LIMIT = 50

# Hard ceiling on tool calls per completion that we CONSIDER for scoring.
# Calls past this index are ignored by the reward function. TRL doesn't
# enforce a hard turn budget natively, so this just clips the transcript.
DEFAULT_MAX_TURNS = 10


def make_run_sql_tool(
    runtimes: DialectRuntime | dict[str, DialectRuntime],
    *,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> Callable[..., dict[str, Any]]:
    """Return a TRL-compatible ``run_sql`` tool bound to one or more runtimes.

    Single-dialect mode (``runtimes`` is a :class:`DialectRuntime`):
        Tool signature ``run_sql(sql_command)``. Closes over the one
        runtime; backwards-compatible with the v0 single-dialect path.

    Multi-dialect mode (``runtimes`` is ``dict[str, DialectRuntime]``):
        Tool signature ``run_sql(sql_command, dialect)``. Dispatches to
        ``runtimes[dialect]`` at call time. The model is expected to
        copy the prompt's "Dialect: X" tag into the ``dialect`` arg
        (the tool-aware system prompt instructs this when built with
        ``with_dialect_arg=True``). Unknown dialect strings come back
        as a structured error in the tool payload rather than raising,
        so the rollout doesn't crash on a malformed assistant turn.

    The function shape is exactly what TRL's tool registry expects: a
    typed signature plus a Google-style docstring. TRL inspects both to
    auto-generate the JSON schema the model sees in its tool list. The
    docstring is built per-mode so the dialect set + the "copy this
    string" rule are visible in the registered schema.

    Args:
        runtimes: A *set-up* :class:`DialectRuntime` (single-dialect)
            or ``dict[str, DialectRuntime]`` (multi-dialect). All
            runtimes must already have ``setup()`` called.
        preview_limit: Maximum rows surfaced back to the model. The
            reward function ignores the preview entirely (it
            re-executes), so this is purely a UX knob.
    """
    if isinstance(runtimes, DialectRuntime):
        return _make_single_run_sql_tool(runtimes, preview_limit=preview_limit)
    if not runtimes:
        raise ValueError("make_run_sql_tool: runtimes dict is empty")
    return _make_dispatch_run_sql_tool(runtimes, preview_limit=preview_limit)


def _make_single_run_sql_tool(
    runtime: DialectRuntime, *, preview_limit: int
) -> Callable[[str], dict[str, Any]]:
    """Single-dialect ``run_sql(sql_command)`` -- the v0 shape."""
    _ = runtime.engine
    _ = runtime.snapshot

    def run_sql(sql_command: str) -> dict[str, Any]:
        run = runtime.run(sql_command)
        return _runresult_to_tool_payload(run, preview_limit=preview_limit)

    run_sql.__doc__ = _build_single_run_sql_docstring(
        dialect=runtime.dialect, preview_limit=preview_limit
    )
    return run_sql


def _make_dispatch_run_sql_tool(
    runtimes: dict[str, DialectRuntime], *, preview_limit: int
) -> Callable[[str, str], dict[str, Any]]:
    """Multi-dialect ``run_sql(sql_command, dialect)`` -- dispatches by arg."""
    for dialect, rt in runtimes.items():
        if not isinstance(rt, DialectRuntime):
            raise TypeError(
                f"runtimes[{dialect!r}] must be a DialectRuntime, got {type(rt).__name__}"
            )
        # Force the property accessors so a forgotten setup() surfaces
        # here, not deep inside a tool call inside a vLLM rollout.
        _ = rt.engine
        _ = rt.snapshot

    known = sorted(runtimes)

    def run_sql(sql_command: str, dialect: str) -> dict[str, Any]:
        rt = runtimes.get(dialect)
        if rt is None:
            return {
                "success": False,
                "row_count": 0,
                "columns": [],
                "rows_preview": [],
                "truncated": False,
                "error": (
                    f"unknown dialect {dialect!r}; valid choices: {known}. "
                    f"Copy the dialect string from the system prompt verbatim."
                ),
                "error_class": "dispatch",
                "execution_time_s": 0.0,
            }
        run = rt.run(sql_command)
        return _runresult_to_tool_payload(run, preview_limit=preview_limit)

    run_sql.__doc__ = _build_dispatch_run_sql_docstring(
        dialects=known, preview_limit=preview_limit
    )
    return run_sql


def _build_single_run_sql_docstring(*, dialect: str, preview_limit: int) -> str:
    return (
        f"Execute a SELECT query in the {dialect} manysql dialect against the "
        f"in-memory catalog.\n\n"
        f"The catalog is the same one described in the system prompt's Schema "
        f"section. The query is parsed by the dialect's grammar and executed "
        f"by the manysql IR engine. A row preview (up to {preview_limit} "
        f"rows) is returned alongside the column list and a total row count. "
        f"Errors are returned (not raised) so you can iterate.\n\n"
        f"Args:\n"
        f"    sql_command: A single SELECT (or WITH ... SELECT) query in the "
        f"{dialect} dialect. No DDL/DML.\n\n"
        f"Returns:\n"
        f"    A dict with keys: success (bool), row_count (int), columns "
        f"(list[str]), rows_preview (list[dict]), truncated (bool), error "
        f"(str | None), error_class (str | None: 'parse'|'runtime'|'empty'), "
        f"execution_time_s (float)."
    )


def _build_dispatch_run_sql_docstring(
    *, dialects: list[str], preview_limit: int
) -> str:
    return (
        f"Execute a SELECT query in the requested manysql dialect against the "
        f"in-memory catalog.\n\n"
        f"This run is a multi-dialect curriculum; each task in the dataset "
        f"specifies which dialect it targets in its system prompt (the "
        f"\"Dialect: X\" tag). Copy that string verbatim into the ``dialect`` "
        f"argument of every call -- the same SQL parsed under a different "
        f"dialect's grammar usually fails or returns different rows.\n\n"
        f"Supported dialects: {dialects}.\n\n"
        f"A row preview (up to {preview_limit} rows) is returned alongside "
        f"the column list and a total row count. Errors are returned (not "
        f"raised) so you can iterate.\n\n"
        f"Args:\n"
        f"    sql_command: A single SELECT (or WITH ... SELECT) query in the "
        f"target dialect. No DDL/DML.\n"
        f"    dialect: One of {dialects}. Match the system prompt's "
        f"        \"Dialect: X\" tag exactly.\n\n"
        f"Returns:\n"
        f"    A dict with keys: success (bool), row_count (int), columns "
        f"(list[str]), rows_preview (list[dict]), truncated (bool), error "
        f"(str | None), error_class (str | None: 'parse'|'runtime'|'empty'|"
        f"'dispatch'), execution_time_s (float)."
    )


def _runresult_to_tool_payload(
    run: Any, *, preview_limit: int
) -> dict[str, Any]:
    """Map a :class:`~train.env.engine.RunResult` to a model-facing dict."""
    er = run.exec_result
    rows = er.rows or []
    truncated = len(rows) > preview_limit
    return {
        "success": er.success,
        "row_count": len(rows),
        "columns": list(er.columns),
        "rows_preview": rows[:preview_limit],
        "truncated": truncated,
        "error": er.error,
        "error_class": run.error_class,
        "execution_time_s": er.execution_time_s,
    }


# ---------------------------------------------------------------------------
# Transcript reconstruction
# ---------------------------------------------------------------------------


def reconstruct_turns(
    completion: list[dict[str, Any]],
    runtime: DialectRuntime,
    *,
    tool_name: str = "run_sql",
    max_turns: int = DEFAULT_MAX_TURNS,
    gold_rows: list[dict[str, Any]] | None = None,
) -> list[Turn]:
    """Walk a TRL completion and rebuild a list of :class:`Turn`.

    A TRL "completion" is a list of message dicts the trainer hands the
    reward function. Assistant turns may carry ``tool_calls``; tool
    turns carry the function's return value as ``content``. We pair
    them up, re-execute each ``sql_command`` against ``runtime`` to get
    full untruncated rows, and emit one ``Turn`` per tool call until
    we hit ``max_turns`` (or run out of calls).

    Re-executing instead of trusting the tool's preview costs O(turns)
    extra dialect calls per generation but buys us:

    * deterministic scoring (the reward only depends on the SQL string);
    * untruncated rows for the row-match comparison even when the model
      saw a truncated preview;
    * decoupling of preview UX (capped) from scoring fidelity (full).

    If ``gold_rows`` is given, the first turn whose result rows match
    is marked ``Turn.matched=True`` and the transcript is truncated
    there (mirrors :class:`SqlEnv` semantics: stop on match). Pass
    ``gold_rows=None`` to skip match-stop-truncation; you'll get every
    tool call up to ``max_turns``.
    """
    turns: list[Turn] = []
    for assistant in _iter_assistant_with_tool_calls(completion):
        if len(turns) >= max_turns:
            break
        for call in assistant.get("tool_calls") or []:
            if len(turns) >= max_turns:
                break
            sql = _extract_sql_arg(call, tool_name=tool_name)
            if sql is None:
                continue
            run = runtime.run(sql)
            er = run.exec_result
            matched = False
            if (
                gold_rows is not None
                and er.success
                and run.error_class is None
            ):
                cmp = compare_results(gold_rows, er.rows)
                matched = cmp.matches
            turns.append(
                Turn(
                    index=len(turns),
                    sql=sql,
                    exec_result=er,
                    error_class=run.error_class,
                    matched=matched,
                )
            )
            if matched:
                # Match-stop: agent solved it; later tool calls are
                # ignored by the reward.
                return turns
    return turns


def _iter_assistant_with_tool_calls(
    completion: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter completion to assistant turns that have actionable SQL.

    Accepts two formats:
    * Structured tool_calls (TRL agent mode with ``tools=``).
    * ``<SQL>...</SQL>`` tag in content (single-turn tag mode, no tools API needed).
    """
    out: list[dict[str, Any]] = []
    for turn in completion:
        if turn.get("role") != "assistant":
            continue
        if turn.get("tool_calls"):
            out.append(turn)
        elif _SQL_TAG_RE.search(turn.get("content") or ""):
            # Synthesise a fake tool_calls entry so the rest of
            # reconstruct_turns can stay format-agnostic.
            sql = _SQL_TAG_RE.search(turn["content"]).group(1).strip()
            synthetic = dict(turn)
            synthetic["tool_calls"] = [
                {"function": {"name": "run_sql", "arguments": {"sql_command": sql}}}
            ]
            out.append(synthetic)
    return out


def _extract_sql_arg(
    call: dict[str, Any], *, tool_name: str
) -> str | None:
    """Pull the ``sql_command`` argument out of one tool call dict."""
    fn = call.get("function") or {}
    if fn.get("name") != tool_name:
        return None
    args = fn.get("arguments")
    # TRL usually pre-parses arguments to a dict, but some serializers
    # leave them as a JSON string. Handle both, fall back to None on
    # malformed payloads (counted as "no SQL emitted" -> skipped turn).
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    sql = args.get("sql_command")
    if not isinstance(sql, str):
        return None
    return sql


# ---------------------------------------------------------------------------
# Reward function builders
# ---------------------------------------------------------------------------


# Components of RewardBreakdown that we expose as independent reward
# functions. Order is the canonical W&B / trackio dashboard order.
REWARD_COMPONENTS: tuple[str, ...] = (
    "correctness",
    "turn_bonus",
    "error_shaping",
    "format_penalty",
    "terminal_penalty",
    "total",
)


def make_reward_funcs(
    runtimes: DialectRuntime | dict[str, DialectRuntime],
    *,
    reward_config: RewardConfig | None = None,
    components: list[str] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    tool_name: str = "run_sql",
) -> list[Callable[..., list[float]]]:
    """Build one TRL-style reward function per ``RewardBreakdown`` field.

    Each returned callable matches TRL's reward-function contract::

        def reward_fn(completions, **dataset_columns) -> list[float]: ...

    GRPOTrainer logs each function under its ``__name__`` independently,
    so the breakdown shows up as N separate panels. Pass the result to
    ``GRPOTrainer(reward_funcs=...)`` directly.

    Multi-dialect: pass a ``dict[dialect, DialectRuntime]`` and the
    reward fn dispatches per row using the dataset's ``dialect``
    column. The model's claimed ``dialect=`` argument in tool calls is
    *ignored* for scoring -- we always re-execute against the row's
    ground-truth dialect, so a model that misuses the dialect arg
    learns from the correctness signal.

    Args:
        runtimes: A single set-up :class:`DialectRuntime` (single-dialect
            mode) or ``dict[str, DialectRuntime]`` (multi-dialect mode).
        reward_config: Forwarded to :func:`compute_reward`. Defaults to
            linear-mode :class:`RewardConfig`.
        components: Subset of :data:`REWARD_COMPONENTS` to expose. Defaults
            to ``["correctness", "turn_bonus", "error_shaping",
            "format_penalty", "terminal_penalty"]`` (every component
            except ``total``, since GRPO sums rewards itself). Pass
            ``["total"]`` if you'd rather log a single scalar.
        max_turns: Hard cap on tool calls considered per completion.
            Must match the value used by the trainer's
            ``max_completion_length`` budget for the metric to be stable.
        tool_name: Name the model uses to invoke SQL execution. Must
            match the function name returned by :func:`make_run_sql_tool`.
    """
    cfg = reward_config or RewardConfig()
    chosen = components or [c for c in REWARD_COMPONENTS if c != "total"]
    _validate_components(chosen)
    runtime_map = _normalize_runtimes(runtimes)

    funcs: list[Callable[..., list[float]]] = []
    for component in chosen:
        funcs.append(
            _make_component_reward_fn(
                component=component,
                runtime_map=runtime_map,
                cfg=cfg,
                max_turns=max_turns,
                tool_name=tool_name,
            )
        )
    return funcs


def _normalize_runtimes(
    runtimes: DialectRuntime | dict[str, DialectRuntime],
) -> dict[str, DialectRuntime]:
    """Coerce single/dict inputs to a uniform dict for internal use."""
    if isinstance(runtimes, DialectRuntime):
        return {runtimes.dialect: runtimes}
    if not runtimes:
        raise ValueError("runtimes dict is empty")
    return dict(runtimes)


def _validate_components(components: list[str]) -> None:
    bad = [c for c in components if c not in REWARD_COMPONENTS]
    if bad:
        raise ValueError(
            f"unknown reward components {bad!r}; valid: {list(REWARD_COMPONENTS)}"
        )


def _make_component_reward_fn(
    *,
    component: str,
    runtime_map: dict[str, DialectRuntime],
    cfg: RewardConfig,
    max_turns: int,
    tool_name: str,
) -> Callable[..., list[float]]:
    """Closure factory: one function per logged component.

    The returned function inspects each row's ``dialect`` column (passed
    in by TRL alongside ``completions``) and looks up the matching
    runtime in ``runtime_map``. In single-dialect mode there's only one
    entry; in multi-dialect mode dispatch happens here so scoring uses
    ground truth regardless of the model's ``dialect=`` tool argument.
    """
    fallback_runtime = next(iter(runtime_map.values()))
    is_multi = len(runtime_map) > 1

    def reward_fn(
        completions: list[list[dict[str, Any]]],
        gold_rows_json: list[str] | None = None,
        dialect: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        if gold_rows_json is None:
            return [0.0] * len(completions)
        if is_multi and dialect is None:
            # Multi-dialect runtime requires the row's dialect to score
            # correctly. Bailing out is safer than silently picking a
            # wrong runtime; the dataset builder always emits ``dialect``
            # so this only fires when the caller has wired the dataset
            # incorrectly.
            return [0.0] * len(completions)
        scores: list[float] = []
        for i, (completion, gold_json) in enumerate(
            zip(completions, gold_rows_json, strict=False)
        ):
            row_dialect = dialect[i] if (dialect and i < len(dialect)) else None
            rt = runtime_map.get(row_dialect, fallback_runtime) if row_dialect else fallback_runtime
            breakdown = score_completion(
                completion=completion,
                gold_rows=json.loads(gold_json),
                runtime=rt,
                cfg=cfg,
                max_turns=max_turns,
                tool_name=tool_name,
            )
            scores.append(getattr(breakdown, component))
        return scores

    reward_fn.__name__ = f"sql_{component}_reward"
    reward_fn.__qualname__ = reward_fn.__name__
    reward_fn.__doc__ = (
        f"GRPO reward function returning the ``{component}`` component "
        f"of :class:`RewardBreakdown` per completion."
    )
    return reward_fn


def score_completion(
    *,
    completion: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    runtime: DialectRuntime,
    cfg: RewardConfig | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    tool_name: str = "run_sql",
) -> RewardBreakdown:
    """Score one TRL completion end-to-end. Public for testing + custom rewards.

    Reconstructs the transcript, builds the final :class:`ComparisonResult`
    against ``gold_rows``, and dispatches to :func:`compute_reward`. Both
    return values (the breakdown) and intermediate steps (the transcript
    via :func:`reconstruct_turns`) are exposed independently so callers
    that want richer custom rewards can compose them.

    Always operates on a single runtime; multi-dialect dispatch is the
    reward-factory's responsibility (see :func:`make_reward_funcs`).
    """
    cfg = cfg or RewardConfig()
    transcript = reconstruct_turns(
        completion=completion,
        runtime=runtime,
        tool_name=tool_name,
        max_turns=max_turns,
        gold_rows=gold_rows,
    )
    final_comparison = None
    if transcript:
        last = transcript[-1]
        if last.exec_result.success and last.error_class is None:
            final_comparison = compare_results(gold_rows, last.exec_result.rows)
    return compute_reward(
        transcript=transcript,
        final_comparison=final_comparison,
        max_turns=max_turns,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def tasks_to_dataset(
    tasks: list[Any],  # list[SqlTask]; Any to keep this module import-cheap
    runtime: DialectRuntime | None = None,
    *,
    system_prompt: str | None = None,
    tokenizer: Any = None,
    max_prompt_tokens: int | None = None,
    extra_columns: dict[str, list[Any]] | None = None,
) -> Dataset:
    """Build a HF :class:`~datasets.Dataset` from a list of :class:`SqlTask`.

    Each row is::

        {
          "prompt":    [{"role": "system", ...}, {"role": "user", ...}],
          "task_id":   <str>,
          "dialect":   <str>,
          "gold_sql":  <str>,
          "gold_rows_json": <json-encoded list of dicts>,
          "generator": <str>,
        }

    ``prompt`` is in chat-message form so GRPOTrainer can apply the
    tokenizer's chat template. ``gold_rows`` is JSON-encoded because the
    underlying Arrow type inference chokes on heterogeneous row schemas
    across tasks (different golden queries return different columns).

    Args:
        tasks: Output of ``some_generator.all_tasks()``.
        runtime: A *set-up* :class:`DialectRuntime`, used only as the
            default source of the system prompt when ``system_prompt``
            isn't given. Optional; if you pass ``system_prompt``
            directly (e.g. in a multi-dialect concat-then-build flow)
            you don't need to pass a runtime at all.
        system_prompt: Override for the system message. Defaults to
            ``runtime.system_prompt()`` (raw-SQL rules); pass
            :func:`trl_agent_system_prompt` output to opt into the
            tool-using agent prompt. Required if ``runtime`` is None.
        tokenizer: Optional HF tokenizer for the prompt-length filter.
            If both ``tokenizer`` and ``max_prompt_tokens`` are given,
            rows whose tokenized prompt exceeds the cap are dropped.
        max_prompt_tokens: See ``tokenizer``.
        extra_columns: Optional dict of column-name -> per-task value
            list, length must equal ``len(tasks)``. Surfaced as kwargs to
            reward functions; useful for piping curriculum tags / seed
            ids through.
    """
    from datasets import Dataset  # noqa: PLC0415  heavy optional dep

    if system_prompt is None:
        if runtime is None:
            raise ValueError(
                "tasks_to_dataset: pass either a runtime (for the default "
                "system prompt) or an explicit system_prompt string"
            )
        sys_prompt = runtime.system_prompt()
    else:
        sys_prompt = system_prompt
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": task.prompt},
                ],
                "task_id": task.meta.task_id,
                "dialect": task.meta.dialect,
                "generator": task.meta.generator,
                "gold_sql": task.gold_sql,
                # JSON-encode to dodge Arrow's struct-schema unification
                # across heterogeneous golden queries.
                "gold_rows_json": json.dumps(task.gold_rows, default=_json_default),
            }
        )
    if extra_columns:
        for k, vs in extra_columns.items():
            if len(vs) != len(rows):
                raise ValueError(
                    f"extra_columns[{k!r}] length {len(vs)} != tasks length {len(rows)}"
                )
            for row, v in zip(rows, vs, strict=False):
                row[k] = v

    ds = Dataset.from_list(rows)

    if tokenizer is not None and max_prompt_tokens is not None:
        def _len(row: dict[str, Any]) -> int:
            return len(
                tokenizer.apply_chat_template(
                    row["prompt"], add_generation_prompt=True, tokenize=True
                )
            )

        ds = ds.filter(lambda r: _len(r) <= max_prompt_tokens)
    return ds


def _json_default(o: Any) -> str:
    """JSON encoder fallback for non-stringifiable cells (datetimes, decimals).

    Polars row dicts can contain ``datetime.date`` / ``datetime.datetime`` /
    ``decimal.Decimal`` values that ``json`` doesn't know how to serialize.
    Stringifying is fine for our purposes: gold rows roundtrip through
    JSON only on the train/score path, and the comparison logic tolerates
    string-vs-numeric divergences via ``numeric_match``.
    """
    return str(o)


# ---------------------------------------------------------------------------
# System prompt helper
# ---------------------------------------------------------------------------


# Replaces the default base rules from DialectRuntime when wiring the
# trainer's tool-using agent. Tells the model to use run_sql instead of
# emitting raw SQL in chat -- otherwise we get nothing parseable in the
# completion.
TRL_AGENT_BASE_RULES = (
    "- You will be given a question (or a SQL translation request) about the data\n"
    "  in the database, and a schema.\n"
    "- Use the `run_sql` tool to execute SELECT queries against the catalog. Do NOT\n"
    "  write SQL in your chat reply.\n"
    "- Each tool call must be exactly one SELECT statement (or one WITH ... SELECT)\n"
    "  in the target dialect. No DDL/DML.\n"
    "- After each tool result, inspect the rows. If the query errored or the rows\n"
    "  look wrong, call the tool again with a corrected query.\n"
    "- When you are confident the result is correct, call `run_sql` ONE more time\n"
    "  with the same final query so the answer is unambiguous.\n"
    "- Use only columns that appear in the schema. Do NOT invent columns.\n"
    "- Add LIMIT to the query when the result could be unbounded; default LIMIT 10.\n"
)


def trl_agent_system_prompt(
    runtime: DialectRuntime, *, with_dialect_arg: bool = False
) -> str:
    """Variant of ``runtime.system_prompt()`` tailored for TRL tool use.

    Same dialect card + schema body, but the rules block tells the
    model to call ``run_sql`` instead of writing SQL in chat. Use this
    in place of :meth:`DialectRuntime.system_prompt` when building the
    GRPO dataset.

    Args:
        runtime: The dialect runtime whose card / schema seed the
            prompt. The current dialect's name is also surfaced as a
            ``Dialect: <name>`` tag at the bottom so the model has an
            unambiguous string to copy into ``dialect=`` calls.
        with_dialect_arg: If True, append a rule telling the model to
            pass ``dialect="<runtime.dialect>"`` to every ``run_sql``
            call. Use this when the trainer's tool was built with the
            multi-dialect dispatch factory (``make_run_sql_tool`` over
            a dict). When False (the default), the tool signature is
            single-arg ``run_sql(sql_command)`` and the dialect tag is
            informational only.
    """
    extra_rule = ""
    if with_dialect_arg:
        extra_rule = (
            f'- ALWAYS pass dialect="{runtime.dialect}" to run_sql so the\n'
            f"  query routes to the correct dialect engine (this run is a\n"
            f"  multi-dialect curriculum -- the same SQL parsed under a\n"
            f"  different grammar usually fails or returns different rows).\n"
        )
    rules = TRL_AGENT_BASE_RULES + extra_rule
    body = runtime.system_prompt(base_rules=rules)
    return f"{body.rstrip()}\n\nDialect: {runtime.dialect}\n"


TRL_TAG_BASE_RULES = (
    "- You will be given a question (or a SQL translation request) about the data\n"
    "  in the database, and a schema.\n"
    "- Write exactly ONE SQL SELECT statement (or WITH ... SELECT) that answers\n"
    "  the question in the target dialect.\n"
    "- Wrap the final SQL between <SQL> and </SQL> tags. Example:\n"
    "    <SQL>SELECT * FROM employees LIMIT 10</SQL>\n"
    "- Only output SQL that is valid in the described dialect. Do NOT invent columns.\n"
    "- Use LIMIT when the result could be unbounded; default LIMIT 10.\n"
)


def trl_tag_system_prompt(runtime: DialectRuntime) -> str:
    """Variant of ``runtime.system_prompt()`` for single-turn tag mode.

    The model writes SQL directly between ``<SQL>...</SQL>`` tags instead
    of calling a tool. Compatible with ``GRPOTrainer`` without the
    ``tools=`` parameter (works with trl <= 0.25.x).
    """
    body = runtime.system_prompt(base_rules=TRL_TAG_BASE_RULES)
    return f"{body.rstrip()}\n\nDialect: {runtime.dialect}\n"


__all__ = [
    "DEFAULT_MAX_TURNS",
    "DEFAULT_PREVIEW_LIMIT",
    "REWARD_COMPONENTS",
    "SQL_TAG_END",
    "SQL_TAG_START",
    "TRL_AGENT_BASE_RULES",
    "TRL_TAG_BASE_RULES",
    "make_reward_funcs",
    "make_run_sql_tool",
    "reconstruct_turns",
    "score_completion",
    "tasks_to_dataset",
    "trl_agent_system_prompt",
    "trl_tag_system_prompt",
]
