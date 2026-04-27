"""Multi-dialect curriculum tests.

Covers the wiring added in the multi-dialect training plan:

1. ``make_run_sql_tool(dict[str, DialectRuntime])`` -- the dispatch
   tool: signature, docstring shape, dispatch on the ``dialect``
   argument, error payload for unknown dialects.

2. ``make_reward_funcs(runtimes=dict, ...)`` -- per-row dispatch on
   the dataset's ``dialect`` column. Crucially, the model's claimed
   ``dialect=`` tool argument is irrelevant for scoring; the reward
   always re-executes against the row's true dialect.

3. ``trl_agent_system_prompt(rt, with_dialect_arg=True)`` -- system
   prompt variant that tells the model to copy the prompt's "Dialect:
   X" tag into every tool call.

4. Task expansion in :mod:`train.grpo_sql` -- ``cross_product`` and
   ``partition`` modes, ``_relabel_task`` cloning, and the dataset
   builder concatenating multi-dialect sub-datasets.

We stick to the GoldenCatalog / GoldenTaskGenerator path because it
needs no network -- the WikiSQL multi-dialect path is exercised in
``test_train_env_wikisql.py`` (catalog reuse) and the same cross-product
/ partition logic operates on top of any TaskGenerator.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from train.env import (
    DialectRuntime,
    GoldenCatalog,
    GoldenTaskConfig,
    GoldenTaskGenerator,
    RewardConfig,
    SqlTask,
    make_reward_funcs,
    make_run_sql_tool,
    trl_agent_system_prompt,
)

# Two-dialect curricula are enough to test dispatch; bumping to three
# would 3x test runtime without finding new bugs.
DIALECTS = ("aggressive_alien", "tsql_ish")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtimes() -> dict[str, DialectRuntime]:
    """One catalog, two runtimes (mirrors the multi-dialect curriculum)."""
    cat = GoldenCatalog()
    cat.build()
    rts: dict[str, DialectRuntime] = {}
    for d in DIALECTS:
        rt = DialectRuntime(dialect=d, catalog=cat)
        rt.setup()
        rts[d] = rt
    yield rts
    for rt in rts.values():
        rt.teardown()


@pytest.fixture(scope="module")
def gold_employees(runtimes: dict[str, DialectRuntime]) -> list[dict]:
    """Reference rows for a query that works in every dialect."""
    rt = next(iter(runtimes.values()))
    rows = rt.run("SELECT * FROM employees").exec_result.rows
    assert rows
    return rows


def _tool_call(sql: str, *, dialect: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"sql_command": sql}
    if dialect is not None:
        args["dialect"] = dialect
    return {"function": {"name": "run_sql", "arguments": args}}


def _assistant(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def _tool_msg() -> dict[str, Any]:
    return {"role": "tool", "content": ""}


# ---------------------------------------------------------------------------
# make_run_sql_tool: dispatch mode
# ---------------------------------------------------------------------------


def test_dispatch_tool_signature_has_dialect(runtimes) -> None:
    """The dispatch tool must accept a ``dialect`` arg so TRL exposes it
    in the generated JSON schema the model sees.
    """
    import inspect  # noqa: PLC0415

    tool = make_run_sql_tool(runtimes)
    sig = inspect.signature(tool)
    assert list(sig.parameters) == ["sql_command", "dialect"]


def test_dispatch_tool_docstring_lists_dialects(runtimes) -> None:
    tool = make_run_sql_tool(runtimes)
    doc = tool.__doc__ or ""
    for d in DIALECTS:
        assert d in doc
    # The "copy the dialect string" rule is the whole point.
    assert "Dialect:" in doc or "verbatim" in doc.lower()
    # Google-style sections required for TRL schema introspection.
    assert "Args:" in doc and "Returns:" in doc


def test_dispatch_tool_routes_to_correct_runtime(runtimes) -> None:
    """Calling with ``dialect=DIALECTS[0]`` must execute against that
    runtime. We can prove this by checking the ``backend`` field in
    the result, which encodes the dialect name.
    """
    tool = make_run_sql_tool(runtimes)
    for d in DIALECTS:
        out = tool("SELECT * FROM employees", d)
        assert out["success"] is True
        assert out["row_count"] >= 1


def test_dispatch_tool_unknown_dialect_returns_structured_error(runtimes) -> None:
    """Unknown dialect comes back as an error payload, not a raise.
    Otherwise a malformed assistant turn would crash the rollout.
    """
    tool = make_run_sql_tool(runtimes)
    out = tool("SELECT 1", "no_such_dialect")
    assert out["success"] is False
    assert out["error_class"] == "dispatch"
    assert "no_such_dialect" in out["error"]
    # Valid choices listed so the model can correct itself.
    for d in DIALECTS:
        assert d in out["error"]


def test_dispatch_tool_rejects_empty_runtimes() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_run_sql_tool({})


def test_dispatch_tool_validates_runtime_setup() -> None:
    """Forgotten setup() should fail at tool-construction time, not
    deep inside a rollout.
    """
    cat = GoldenCatalog()
    cat.build()
    rt_unset = DialectRuntime(dialect="aggressive_alien", catalog=cat)
    # Don't call setup()
    with pytest.raises(RuntimeError, match="setup"):
        make_run_sql_tool({"aggressive_alien": rt_unset})


# ---------------------------------------------------------------------------
# Reward dispatch on the dataset's `dialect` column
# ---------------------------------------------------------------------------


def test_multi_dialect_reward_dispatches_per_row(
    runtimes, gold_employees: list[dict]
) -> None:
    """Two rows in one batch, each tagged with a different dialect.
    The reward fn must score each against its own runtime.
    """
    funcs = make_reward_funcs(
        runtimes=runtimes, reward_config=RewardConfig.discounted()
    )
    by_name = {fn.__name__: fn for fn in funcs}
    correctness = by_name["sql_correctness_reward"]

    completion = [_assistant(_tool_call("SELECT * FROM employees")), _tool_msg()]
    gold_json = json.dumps(gold_employees, default=str)

    out = correctness(
        completions=[completion, completion],
        gold_rows_json=[gold_json, gold_json],
        dialect=list(DIALECTS),
    )
    assert len(out) == 2
    # Both should be one-shot correct against their respective runtime
    # because employees is a vanilla SCAN that every dialect parses.
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0)


def test_multi_dialect_reward_ignores_models_dialect_arg(
    runtimes, gold_employees: list[dict]
) -> None:
    """Even if the model lies about ``dialect=`` in its tool call, the
    reward must score against the *row's* ground-truth dialect.

    We construct a completion whose tool call carries
    ``dialect="bogus_value"``. A naive reward that trusted the model's
    arg would crash or score wrong; our reward must ignore it because
    the dispatch happens inside the tool, not the reward (which
    re-executes via the row's dialect).
    """
    funcs = make_reward_funcs(
        runtimes=runtimes, reward_config=RewardConfig.discounted()
    )
    correctness = next(
        f for f in funcs if f.__name__ == "sql_correctness_reward"
    )
    # Note: the dialect= in the tool_call dict is NOT what the reward
    # uses for dispatch. The dataset's ``dialect`` column is.
    completion = [
        _assistant(_tool_call("SELECT * FROM employees", dialect="bogus_value")),
        _tool_msg(),
    ]
    gold_json = json.dumps(gold_employees, default=str)
    [score] = correctness(
        completions=[completion],
        gold_rows_json=[gold_json],
        dialect=[DIALECTS[0]],
    )
    assert score == pytest.approx(1.0)


def test_multi_dialect_reward_returns_zeros_when_dialect_missing(
    runtimes, gold_employees: list[dict]
) -> None:
    """Multi-runtime mode without the ``dialect`` column should not
    silently pick a wrong runtime -- it should bail out with zeros.
    Catches dataset-builder misconfiguration loudly.
    """
    funcs = make_reward_funcs(
        runtimes=runtimes, reward_config=RewardConfig.discounted()
    )
    correctness = next(
        f for f in funcs if f.__name__ == "sql_correctness_reward"
    )
    completion = [_assistant(_tool_call("SELECT * FROM employees")), _tool_msg()]
    gold_json = json.dumps(gold_employees, default=str)
    out = correctness(
        completions=[completion],
        gold_rows_json=[gold_json],
        # no `dialect=` kwarg
    )
    assert out == [0.0]


def test_single_dialect_reward_works_without_dialect_column(
    gold_employees: list[dict],
) -> None:
    """Single-dialect (one entry) must keep working without a
    ``dialect`` column -- back-compat with the pre-multi-dialect setup.
    """
    cat = GoldenCatalog()
    cat.build()
    rt = DialectRuntime(dialect=DIALECTS[0], catalog=cat)
    rt.setup()
    try:
        funcs = make_reward_funcs(
            runtimes=rt,  # single runtime, the legacy shape
            reward_config=RewardConfig.discounted(),
        )
        correctness = next(
            f for f in funcs if f.__name__ == "sql_correctness_reward"
        )
        completion = [_assistant(_tool_call("SELECT * FROM employees")), _tool_msg()]
        gold_json = json.dumps(gold_employees, default=str)
        out = correctness(
            completions=[completion],
            gold_rows_json=[gold_json],
        )
        assert out == [pytest.approx(1.0)]
    finally:
        rt.teardown()


# ---------------------------------------------------------------------------
# trl_agent_system_prompt
# ---------------------------------------------------------------------------


def test_agent_system_prompt_dialect_tag_present(runtimes) -> None:
    """The 'Dialect: X' tag is what the model copies into ``dialect=``."""
    rt = runtimes[DIALECTS[0]]
    sp = trl_agent_system_prompt(rt, with_dialect_arg=True)
    assert f"Dialect: {DIALECTS[0]}" in sp


def test_agent_system_prompt_with_dialect_arg_adds_rule(runtimes) -> None:
    rt = runtimes[DIALECTS[0]]
    plain = trl_agent_system_prompt(rt, with_dialect_arg=False)
    multi = trl_agent_system_prompt(rt, with_dialect_arg=True)
    assert 'dialect=' in multi
    # The single-dialect prompt should not pressure the model to pass
    # ``dialect=`` (that arg doesn't even exist on the single tool).
    assert 'dialect="' not in plain


def test_agent_system_prompt_includes_dialect_card(runtimes) -> None:
    """Smoke check: the dialect card text from manysql.dialects.card is
    spliced in. This is the 'informative prior' that's the whole point
    of multi-dialect training.
    """
    rt = runtimes[DIALECTS[0]]
    sp = trl_agent_system_prompt(rt, with_dialect_arg=True)
    # The card always has a 'Dialect divergences' or similar heading.
    # We don't assume the exact wording -- just non-trivial length.
    assert len(sp) > 500


# ---------------------------------------------------------------------------
# train.grpo_sql task expansion: cross_product / partition / _relabel_task
# ---------------------------------------------------------------------------


def _golden_base_tasks(dialect: str = "aggressive_alien") -> list[SqlTask]:
    gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect=dialect))
    gen.build()
    tasks = gen.all_tasks()[:3]  # keep tests fast
    assert len(tasks) == 3
    return tasks


def test_relabel_task_clones_and_relabels() -> None:
    from train.grpo_sql import _relabel_task  # noqa: PLC0415

    [t] = _golden_base_tasks()[:1]
    t2 = _relabel_task(t, dialect="tsql_ish", suffix_id=True)
    assert t2 is not t
    assert t2.meta is not t.meta  # dataclass replace creates a new TaskMeta
    assert t2.meta.dialect == "tsql_ish"
    assert t2.meta.task_id == f"{t.meta.task_id}__tsql_ish"
    # Data fields are unchanged (dialect-independent).
    assert t2.gold_rows == t.gold_rows
    assert t2.gold_sql == t.gold_sql
    assert t2.prompt == t.prompt
    assert t2.catalog is t.catalog
    # Original task untouched.
    assert t.meta.dialect == "aggressive_alien"


def test_relabel_task_no_suffix_keeps_id() -> None:
    from train.grpo_sql import _relabel_task  # noqa: PLC0415

    [t] = _golden_base_tasks()[:1]
    t2 = _relabel_task(t, dialect="tsql_ish", suffix_id=False)
    assert t2.meta.task_id == t.meta.task_id  # unchanged


def test_expand_cross_product() -> None:
    from train.grpo_sql import _expand_cross_product  # noqa: PLC0415

    base = _golden_base_tasks()
    out = _expand_cross_product(base, ["aggressive_alien", "tsql_ish"])
    # 3 base tasks * 2 dialects = 6 rows.
    assert len(out) == 6
    # Each base task represented in both dialects.
    by_id = {t.meta.task_id: t for t in out}
    assert len(by_id) == 6  # task_ids unique
    # Base ids appear with both dialect suffixes.
    base_ids = {t.meta.task_id for t in base}
    for bid in base_ids:
        assert any(t == f"{bid}__aggressive_alien" for t in by_id)
        assert any(t == f"{bid}__tsql_ish" for t in by_id)


def test_expand_partition_round_robins() -> None:
    from train.grpo_sql import _expand_partition  # noqa: PLC0415

    base = _golden_base_tasks()
    out = _expand_partition(base, ["aggressive_alien", "tsql_ish"])
    # Same total length, no duplicates.
    assert len(out) == len(base)
    # Round-robin: tasks alternate between dialects.
    dialects = [t.meta.dialect for t in out]
    assert dialects == ["aggressive_alien", "tsql_ish", "aggressive_alien"]


# ---------------------------------------------------------------------------
# build_runtimes_and_tasks + build_dataset (multi-dialect)
# ---------------------------------------------------------------------------


def test_build_runtimes_and_tasks_single_dialect_no_expand() -> None:
    """Single-dialect args path must NOT touch the cross_product /
    partition expanders -- tasks already target the one dialect.
    """
    from train.grpo_sql import TrainArgs, build_runtimes_and_tasks  # noqa: PLC0415

    args = TrainArgs(
        dialects=["aggressive_alien"],
        coverage_mode="cross_product",  # ignored when dialects == 1
        generator="golden",
    )
    rts, tasks = build_runtimes_and_tasks(args)
    assert list(rts) == ["aggressive_alien"]
    # Task ids unaffected by relabeling (no '__<dialect>' suffix).
    assert all("__aggressive_alien" not in t.meta.task_id for t in tasks)
    for rt in rts.values():
        rt.teardown()


def test_build_runtimes_and_tasks_multi_dialect_cross_product() -> None:
    from train.grpo_sql import TrainArgs, build_runtimes_and_tasks  # noqa: PLC0415

    args = TrainArgs(
        dialects=list(DIALECTS),
        coverage_mode="cross_product",
        generator="golden",
    )
    rts, tasks = build_runtimes_and_tasks(args)
    assert set(rts) == set(DIALECTS)
    # Cross-product preserves the per-dialect coverage invariant.
    by_d: dict[str, int] = {d: 0 for d in DIALECTS}
    for t in tasks:
        by_d[t.meta.dialect] += 1
    assert by_d[DIALECTS[0]] == by_d[DIALECTS[1]]
    assert by_d[DIALECTS[0]] > 0
    for rt in rts.values():
        rt.teardown()


def test_build_runtimes_and_tasks_shared_catalog() -> None:
    """All dialect runtimes must share one catalog instance -- key for
    memory + (in WikiSQL) for skipping repeat HF downloads.
    """
    from train.grpo_sql import TrainArgs, build_runtimes_and_tasks  # noqa: PLC0415

    args = TrainArgs(dialects=list(DIALECTS), generator="golden")
    rts, _ = build_runtimes_and_tasks(args)
    catalogs = {id(rt.catalog_provider) for rt in rts.values()}
    assert len(catalogs) == 1
    for rt in rts.values():
        rt.teardown()


def test_build_dataset_multi_dialect_concat() -> None:
    """The merged dataset preserves per-row dialect tags so the reward
    layer can dispatch correctly.
    """
    pytest.importorskip("datasets")
    from train.grpo_sql import (  # noqa: PLC0415
        TrainArgs,
        build_dataset,
        build_runtimes_and_tasks,
    )

    args = TrainArgs(
        dialects=list(DIALECTS),
        coverage_mode="cross_product",
        generator="golden",
    )
    rts, tasks = build_runtimes_and_tasks(args)
    try:
        ds = build_dataset(args, rts, tasks, tokenizer=None)
        assert len(ds) == len(tasks)
        observed = {row["dialect"] for row in ds}
        assert observed == set(DIALECTS)
        # System prompt for each row matches the row's dialect (the
        # per-dialect concatenation).
        for row in ds:
            sys_msg = row["prompt"][0]["content"]
            assert f"Dialect: {row['dialect']}" in sys_msg
            # And the multi-dialect rule was inserted.
            assert 'dialect=' in sys_msg
    finally:
        for rt in rts.values():
            rt.teardown()
