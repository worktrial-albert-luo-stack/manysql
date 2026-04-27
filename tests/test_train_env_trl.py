"""Tests for the TRL adapter in ``train.env.trl``.

The adapter is what bridges our env into ``trl.GRPOTrainer``'s agent
mode. It has three independent surfaces that we test in isolation:

1.  ``make_run_sql_tool`` -- closure-bound ``run_sql`` whose return
    value is the model-facing tool payload (capped row preview, error
    fields, error class). Verified for happy path + parse error.
2.  ``reconstruct_turns`` -- walks a TRL-shaped completion (list of
    role/tool_calls dicts) and rebuilds the ``Turn`` list by
    re-executing each ``sql_command``. Verified for: extracted SQL,
    error-class tagging, dict-vs-string argument payloads, match-stop
    truncation, max_turns truncation, missing/malformed args.
3.  ``make_reward_funcs`` + ``score_completion`` -- per-component
    reward functions that GRPOTrainer logs as separate metrics.
    Verified for: function names + count, scalar list shape, end-to-
    end match-on-turn-1 hits the discount-mode unit reward.

We also exercise ``tasks_to_dataset`` if ``datasets`` is installed
(skipped otherwise). The training-side wiring (Unsloth/vLLM/TRL) is
not testable on CPU; ``train/grpo_sql.py --dry-run`` is the integration
smoke test for that path.

Like the rest of ``test_train_env.py`` we use ``aggressive_alien`` as
the canary dialect; it has the heaviest surface divergence so if the
adapter works there it works everywhere.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from train.env import (
    DialectRuntime,
    GoldenCatalog,
    GoldenTaskConfig,
    GoldenTaskGenerator,
    RewardConfig,
    SqlTask,
)
from train.env.rewards import RewardBreakdown
from train.env.trl import (
    DEFAULT_MAX_TURNS,
    REWARD_COMPONENTS,
    TRL_AGENT_BASE_RULES,
    make_reward_funcs,
    make_run_sql_tool,
    reconstruct_turns,
    score_completion,
    tasks_to_dataset,
    trl_agent_system_prompt,
)

DIALECT = "aggressive_alien"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime() -> DialectRuntime:
    rt = DialectRuntime(dialect=DIALECT, catalog=GoldenCatalog())
    rt.setup()
    yield rt
    rt.teardown()


@pytest.fixture(scope="module")
def gold_employees(runtime: DialectRuntime) -> list[dict]:
    """Reference rows for the employees scan: the canonical 'easy task'."""
    rows = runtime.run("SELECT * FROM employees").exec_result.rows
    assert rows, "employees catalog should have rows"
    return rows


def _tool_call(sql: str, *, name: str = "run_sql") -> dict:
    """Build one TRL tool-call dict for use in fake completions."""
    return {
        "function": {
            "name": name,
            "arguments": {"sql_command": sql},
        }
    }


def _assistant(*tool_calls: dict, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(tool_calls)}


def _tool_msg(payload: dict | str = "") -> dict:
    """A tool-role response. Content is opaque to the adapter (it
    re-executes), so we can stuff anything in here.
    """
    return {"role": "tool", "content": payload}


# ---------------------------------------------------------------------------
# make_run_sql_tool
# ---------------------------------------------------------------------------


def test_make_run_sql_tool_requires_setup() -> None:
    rt = DialectRuntime(dialect=DIALECT, catalog=GoldenCatalog())
    # Don't call setup(); accessing engine via the factory should raise.
    with pytest.raises(RuntimeError, match="setup"):
        make_run_sql_tool(rt)


def test_run_sql_tool_happy_path_returns_preview(runtime: DialectRuntime) -> None:
    tool = make_run_sql_tool(runtime, preview_limit=3)
    out = tool("SELECT * FROM employees")
    assert out["success"] is True
    assert out["error"] is None
    assert out["error_class"] is None
    assert out["row_count"] >= 1
    # The preview is capped at preview_limit while row_count stays full.
    assert len(out["rows_preview"]) <= 3
    if out["row_count"] > 3:
        assert out["truncated"] is True
    assert "name" in out["columns"] or out["columns"]  # any column will do
    assert isinstance(out["execution_time_s"], float)


def test_run_sql_tool_classifies_parse_error(runtime: DialectRuntime) -> None:
    tool = make_run_sql_tool(runtime)
    out = tool("totally not sql")
    assert out["success"] is False
    assert out["error_class"] == "parse"
    assert out["error"]
    assert out["row_count"] == 0
    assert out["rows_preview"] == []


def test_run_sql_tool_classifies_empty_sql(runtime: DialectRuntime) -> None:
    tool = make_run_sql_tool(runtime)
    out = tool("   ")
    assert out["success"] is False
    assert out["error_class"] == "empty"


def test_run_sql_tool_docstring_mentions_dialect(runtime: DialectRuntime) -> None:
    tool = make_run_sql_tool(runtime, preview_limit=7)
    assert tool.__doc__
    assert DIALECT in tool.__doc__
    # Google-style sections must be present for TRL's schema introspection.
    assert "Args:" in tool.__doc__
    assert "Returns:" in tool.__doc__
    assert "sql_command" in tool.__doc__
    # Preview limit gets baked in so the model knows.
    assert "7" in tool.__doc__


# ---------------------------------------------------------------------------
# reconstruct_turns
# ---------------------------------------------------------------------------


def test_reconstruct_turns_happy_path_one_call(runtime: DialectRuntime) -> None:
    completion = [
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 1
    assert turns[0].sql == "SELECT * FROM employees"
    assert turns[0].error_class is None
    assert turns[0].exec_result.success
    # match=False because we didn't pass gold_rows.
    assert turns[0].matched is False


def test_reconstruct_turns_tags_parse_and_runtime(runtime: DialectRuntime) -> None:
    completion = [
        _assistant(_tool_call("totally not sql")),
        _tool_msg(""),
        _assistant(_tool_call("SELECT nonexistent_col FROM employees")),
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 2
    assert turns[0].error_class == "parse"
    # Lowering / runtime errors share the 'runtime' tag.
    assert turns[1].error_class in {"runtime", "parse"}  # depends on grammar


def test_reconstruct_turns_match_stop_truncates(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    """If a turn matches gold rows, later tool calls in the same
    completion are dropped (mirrors SqlEnv stop-on-match semantics).
    """
    completion = [
        _assistant(_tool_call("totally not sql")),
        _tool_msg(""),
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
        _assistant(_tool_call("SELECT * FROM employees LIMIT 1")),
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime, gold_rows=gold_employees)
    assert len(turns) == 2  # truncated at the matching turn
    assert turns[0].error_class == "parse"
    assert turns[1].matched is True


def test_reconstruct_turns_respects_max_turns(runtime: DialectRuntime) -> None:
    completion = [
        _assistant(_tool_call("totally not sql")),
        _tool_msg(""),
        _assistant(_tool_call("more not sql")),
        _tool_msg(""),
        _assistant(_tool_call("still not sql")),
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime, max_turns=2)
    assert len(turns) == 2


def test_reconstruct_turns_handles_string_arguments(runtime: DialectRuntime) -> None:
    """Some serializers leave tool-call arguments as a JSON string instead
    of a dict. The adapter must roundtrip-decode them.
    """
    completion = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_sql",
                        "arguments": json.dumps({"sql_command": "SELECT 1"}),
                    }
                }
            ],
        },
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 1
    assert turns[0].sql == "SELECT 1"


def test_reconstruct_turns_skips_malformed_calls(runtime: DialectRuntime) -> None:
    """Tool calls without sql_command, with the wrong tool name, or with
    malformed JSON args are all silently skipped (counted as 'no SQL
    emitted'). Free-text assistant turns are also skipped.
    """
    completion = [
        # wrong tool name
        _assistant(_tool_call("SELECT 1", name="not_run_sql")),
        # missing sql_command argument
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_sql", "arguments": {"foo": "bar"}}}
            ],
        },
        # malformed JSON string args
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_sql", "arguments": "not json {{"}}
            ],
        },
        # free text, no tool_calls at all
        {"role": "assistant", "content": "I think the answer is 42."},
        # one valid call mixed in
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
    ]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 1
    assert "employees" in turns[0].sql.lower()


def test_reconstruct_turns_empty_completion(runtime: DialectRuntime) -> None:
    """A completion with no tool calls at all returns no turns."""
    completion = [{"role": "assistant", "content": "I refuse to query anything."}]
    turns = reconstruct_turns(completion, runtime)
    assert turns == []


# ---------------------------------------------------------------------------
# <SQL> tag extraction (tag-mode reward path)
# ---------------------------------------------------------------------------


def test_reconstruct_turns_tag_mode_picks_last_sql_block(
    runtime: DialectRuntime,
) -> None:
    """Tag mode: the LAST <SQL>...</SQL> in the reply is the answer.

    A model that drafts SQL inside its reasoning and then commits to a
    different final query must be scored against the final query, not
    the draft. This matches ``eval.prompt.extract_sql`` so train and
    eval extractors agree.
    """
    content = (
        "Let me think. A first draft might be:\n"
        "<SQL>SELECT 1 FROM nope_invented_table</SQL>\n"
        "But that table isn't in the schema. The right answer is:\n"
        "<SQL>SELECT * FROM employees</SQL>"
    )
    completion = [{"role": "assistant", "content": content}]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 1
    assert "employees" in turns[0].sql.lower()
    assert "nope_invented_table" not in turns[0].sql.lower()
    # The final tag is valid SQL and runs cleanly.
    assert turns[0].exec_result.success is True
    assert turns[0].error_class is None


def test_reconstruct_turns_tag_mode_tolerates_whitespace_in_tags(
    runtime: DialectRuntime,
) -> None:
    """``< SQL >`` (with internal whitespace) extracts the same as ``<SQL>``.

    The eval-side extractor is whitespace-tolerant; train must match so
    a model that emits ``< SQL >`` is scored, not silently ignored.
    """
    content = "Here's the answer:\n< SQL >SELECT * FROM employees< / SQL >"
    completion = [{"role": "assistant", "content": content}]
    turns = reconstruct_turns(completion, runtime)
    assert len(turns) == 1
    assert "employees" in turns[0].sql.lower()
    assert turns[0].exec_result.success is True


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def test_make_reward_funcs_default_components(runtime: DialectRuntime) -> None:
    funcs = make_reward_funcs(runtimes=runtime, reward_config=RewardConfig())
    names = [f.__name__ for f in funcs]
    # Default = every component except `total` (GRPO sums rewards itself).
    assert names == [
        "sql_correctness_reward",
        "sql_turn_bonus_reward",
        "sql_error_shaping_reward",
        "sql_format_penalty_reward",
        "sql_terminal_penalty_reward",
    ]
    for fn in funcs:
        assert fn.__doc__


def test_make_reward_funcs_custom_components(runtime: DialectRuntime) -> None:
    funcs = make_reward_funcs(
        runtimes=runtime, components=["total", "correctness"]
    )
    names = [f.__name__ for f in funcs]
    assert names == ["sql_total_reward", "sql_correctness_reward"]


def test_make_reward_funcs_rejects_unknown_component(
    runtime: DialectRuntime,
) -> None:
    with pytest.raises(ValueError, match="unknown reward components"):
        make_reward_funcs(runtimes=runtime, components=["bogus"])


def test_reward_components_constant_matches_breakdown_fields() -> None:
    """Guard: if RewardBreakdown adds a field, REWARD_COMPONENTS should
    too -- otherwise that component won't be loggable.
    """
    breakdown_fields = {f.name for f in fields(RewardBreakdown)}
    assert breakdown_fields == set(REWARD_COMPONENTS)


def test_reward_funcs_zero_when_gold_rows_missing(runtime: DialectRuntime) -> None:
    """Reward fns must return zeros (not crash) if the dataset wasn't
    built with our schema. Avoids silently rewarding random SQL.
    """
    fn = make_reward_funcs(runtimes=runtime, components=["total"])[0]
    out = fn(prompts=[[]] * 2, completions=[[], []])
    assert out == [0.0, 0.0]


def test_reward_funcs_correct_sql_one_shot_unit_reward(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    """Discounted mode, one-shot match: correctness == discount^0 == 1.0."""
    funcs = make_reward_funcs(
        runtimes=runtime,
        reward_config=RewardConfig.discounted(),
        max_turns=3,
    )
    by_name = {fn.__name__: fn for fn in funcs}
    completion = [
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
    ]
    gold_json = json.dumps(gold_employees, default=str)
    correctness = by_name["sql_correctness_reward"](
        completions=[completion],
        gold_rows_json=[gold_json],
    )
    assert correctness == [pytest.approx(1.0)]
    # No errors, so error_shaping is 0.
    error_shaping = by_name["sql_error_shaping_reward"](
        completions=[completion],
        gold_rows_json=[gold_json],
    )
    assert error_shaping == [pytest.approx(0.0)]
    # Single turn match => no terminal penalty.
    terminal = by_name["sql_terminal_penalty_reward"](
        completions=[completion],
        gold_rows_json=[gold_json],
    )
    assert terminal == [pytest.approx(0.0)]


def test_reward_funcs_decay_with_turn_index(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    """Discounted mode, recovery on turn 3: correctness == 0.9^2 == 0.81."""
    completion = [
        _assistant(_tool_call("totally not sql")),
        _tool_msg(""),
        _assistant(_tool_call("still not sql")),
        _tool_msg(""),
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
    ]
    fn = next(
        f
        for f in make_reward_funcs(
            runtimes=runtime, reward_config=RewardConfig.discounted()
        )
        if f.__name__ == "sql_correctness_reward"
    )
    [correctness] = fn(
        completions=[completion],
        gold_rows_json=[json.dumps(gold_employees, default=str)],
    )
    assert correctness == pytest.approx(0.81, abs=1e-6)


def test_reward_funcs_batch_independence(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    """A batch with mixed-quality completions should produce one score per
    completion, in order.
    """
    fn = next(
        f
        for f in make_reward_funcs(
            runtimes=runtime, reward_config=RewardConfig.discounted()
        )
        if f.__name__ == "sql_correctness_reward"
    )
    good = [_assistant(_tool_call("SELECT * FROM employees")), _tool_msg("")]
    bad = [_assistant(_tool_call("not sql")), _tool_msg("")]
    silent: list[dict] = []
    gold_json = json.dumps(gold_employees, default=str)
    out = fn(
        completions=[good, bad, silent, good],
        gold_rows_json=[gold_json] * 4,
    )
    assert len(out) == 4
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_completion (whole-breakdown helper)
# ---------------------------------------------------------------------------


def test_score_completion_first_turn_match(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    completion = [
        _assistant(_tool_call("SELECT * FROM employees")),
        _tool_msg(""),
    ]
    bd = score_completion(
        completion=completion,
        gold_rows=gold_employees,
        runtime=runtime,
        cfg=RewardConfig(),
        max_turns=3,
    )
    assert bd.correctness == pytest.approx(5.0)
    assert bd.turn_bonus == pytest.approx(2.0)
    assert bd.error_shaping == 0.0
    assert bd.terminal_penalty == 0.0


def test_score_completion_truncated_with_parse_errors(
    runtime: DialectRuntime, gold_employees: list[dict]
) -> None:
    """Hits the budget with parse errors throughout: every penalty fires."""
    completion = [
        _assistant(_tool_call("totally not sql")),
        _tool_msg(""),
        _assistant(_tool_call("still not sql")),
        _tool_msg(""),
    ]
    bd = score_completion(
        completion=completion,
        gold_rows=gold_employees,
        runtime=runtime,
        cfg=RewardConfig(),
        max_turns=2,
    )
    assert bd.correctness == 0.0
    assert bd.turn_bonus == 0.0
    assert bd.error_shaping == pytest.approx(-2.0)  # -1 per parse turn
    assert bd.format_penalty == pytest.approx(-1.0)
    assert bd.terminal_penalty == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Dataset builder (skipped if `datasets` isn't installed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_tasks() -> list[SqlTask]:
    gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect=DIALECT))
    gen.build()
    tasks = gen.all_tasks()
    assert tasks
    return tasks


def test_tasks_to_dataset_round_trip(
    runtime: DialectRuntime, golden_tasks: list[SqlTask]
) -> None:
    pytest.importorskip("datasets")
    ds = tasks_to_dataset(tasks=golden_tasks[:3], runtime=runtime)
    assert len(ds) == 3
    row = ds[0]
    # Core columns the trainer + reward functions expect.
    for col in (
        "prompt", "task_id", "dialect", "generator", "gold_sql", "gold_rows_json"
    ):
        assert col in row, f"missing column: {col}"
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    # gold_rows_json roundtrips to JSON.
    assert isinstance(row["gold_rows_json"], str)
    json.loads(row["gold_rows_json"])  # must not raise


def test_tasks_to_dataset_uses_agent_system_prompt_when_provided(
    runtime: DialectRuntime, golden_tasks: list[SqlTask]
) -> None:
    pytest.importorskip("datasets")
    ds = tasks_to_dataset(
        tasks=golden_tasks[:1],
        runtime=runtime,
        system_prompt=trl_agent_system_prompt(runtime),
    )
    sys_msg = ds[0]["prompt"][0]["content"]
    assert "run_sql" in sys_msg
    # Falls back to runtime.system_prompt() (raw-SQL rules) when no
    # override is provided.
    ds_default = tasks_to_dataset(tasks=golden_tasks[:1], runtime=runtime)
    default_sys = ds_default[0]["prompt"][0]["content"]
    assert "run_sql" not in default_sys


def test_tasks_to_dataset_extra_columns_threaded_through(
    runtime: DialectRuntime, golden_tasks: list[SqlTask]
) -> None:
    pytest.importorskip("datasets")
    tasks = golden_tasks[:2]
    ds = tasks_to_dataset(
        tasks=tasks,
        runtime=runtime,
        extra_columns={"difficulty": ["easy", "hard"]},
    )
    assert ds[0]["difficulty"] == "easy"
    assert ds[1]["difficulty"] == "hard"


def test_tasks_to_dataset_extra_columns_length_mismatch_raises(
    runtime: DialectRuntime, golden_tasks: list[SqlTask]
) -> None:
    pytest.importorskip("datasets")
    with pytest.raises(ValueError, match="length"):
        tasks_to_dataset(
            tasks=golden_tasks[:2],
            runtime=runtime,
            extra_columns={"x": ["only-one"]},
        )


# ---------------------------------------------------------------------------
# trl_agent_system_prompt
# ---------------------------------------------------------------------------


def test_trl_agent_system_prompt_mentions_tool(runtime: DialectRuntime) -> None:
    sp = trl_agent_system_prompt(runtime)
    assert "run_sql" in sp
    assert "Schema:" in sp  # dialect runtime composition still present
    assert TRL_AGENT_BASE_RULES.split("\n", 1)[0].strip().lstrip("- ") in sp


# ---------------------------------------------------------------------------
# Sanity: defaults stay reasonable
# ---------------------------------------------------------------------------


def test_default_max_turns_is_positive() -> None:
    assert DEFAULT_MAX_TURNS > 0
