"""End-to-end tests for the SQL RL env in ``train.env``.

Coverage map:

* :mod:`train.env.engine`   -- DialectRuntime parse / lower / execute happy
                                path + parse-error + runtime-error tagging.
* :mod:`train.env.catalog`  -- GoldenCatalog snapshot shape.
* :mod:`train.env.tasks`    -- GoldenTaskGenerator builds tasks with
                                non-empty gold rows.
* :mod:`train.env.sql_env`  -- reset/step/done lifecycle, observation
                                strings, episode-result wiring.
* :mod:`train.env.rewards`  -- correctness on first try, partial credit,
                                turn bonus decay, error shaping, format
                                penalty.
* :mod:`train.env.rollout`  -- single- and multi-turn rollouts via
                                FixedSqlPolicy.

Tests use ``aggressive_alien`` as the canary dialect: it has the
heaviest surface divergence from the reference, so if its parser
+ lowering + executor work end-to-end, the env stack is wired right.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from eval.executors.base import ExecResult
from eval.validator import ComparisonResult
from train.env import (
    CatalogSnapshot,
    DialectRuntime,
    EpisodeResult,
    FixedSqlPolicy,
    GoldenCatalog,
    GoldenTaskConfig,
    GoldenTaskGenerator,
    RewardBreakdown,
    RewardConfig,
    SqlEnv,
    SqlTask,
    TaskMeta,
    Turn,
    compute_reward,
    run_episode,
)
from train.env.types import StepResult

DIALECT = "aggressive_alien"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_golden_catalog_build_returns_full_snapshot() -> None:
    snap = GoldenCatalog().build()
    assert isinstance(snap, CatalogSnapshot)
    # All five canonical tables present.
    for tbl in ("employees", "departments", "regions", "sales", "categories"):
        assert tbl in snap.tables
        assert tbl in snap.schemas
        assert isinstance(snap.tables[tbl], pl.DataFrame)
        assert snap.schemas[tbl]
    assert "Tables" in snap.schema_prompt


# ---------------------------------------------------------------------------
# DialectRuntime
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime() -> DialectRuntime:
    rt = DialectRuntime(dialect=DIALECT, catalog=GoldenCatalog())
    rt.setup()
    yield rt
    rt.teardown()


def test_runtime_run_simple_select_succeeds(runtime: DialectRuntime) -> None:
    res = runtime.run("SELECT * FROM employees")
    assert res.error_class is None, res.exec_result.error
    assert res.exec_result.success
    assert len(res.exec_result.rows) == 8
    # Backend label encodes the dialect for downstream attribution.
    assert res.exec_result.backend == f"manysql:{DIALECT}"


def test_runtime_run_classifies_parse_error(runtime: DialectRuntime) -> None:
    res = runtime.run("totally not sql")
    assert not res.exec_result.success
    assert res.error_class == "parse"


def test_runtime_run_classifies_runtime_error(runtime: DialectRuntime) -> None:
    res = runtime.run("SELECT * FROM no_such_table")
    assert not res.exec_result.success
    assert res.error_class == "runtime"


def test_runtime_run_handles_empty_sql(runtime: DialectRuntime) -> None:
    res = runtime.run("   ;  ")
    assert not res.exec_result.success
    assert res.error_class == "empty"


def test_runtime_system_prompt_contains_dialect_card_and_schema(runtime: DialectRuntime) -> None:
    sp = runtime.system_prompt()
    assert "aggressive_alien" in sp
    assert "Tables" in sp


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_tasks() -> list[SqlTask]:
    gen = GoldenTaskGenerator(GoldenTaskConfig(target_dialect=DIALECT))
    gen.build()
    return gen.all_tasks()


def test_golden_generator_emits_tasks_with_gold_rows(golden_tasks: list[SqlTask]) -> None:
    assert golden_tasks, "GoldenTaskGenerator should yield at least one task"
    # Every emitted task came from the reference dialect succeeding, so it
    # must carry non-empty gold metadata.
    for task in golden_tasks:
        assert task.meta.dialect == DIALECT
        assert task.meta.generator == "golden"
        assert task.gold_sql
        # Some queries are 0-row by construction (e.g. filters that exclude
        # everything); we only require the field to exist and be a list.
        assert isinstance(task.gold_rows, list)
        assert task.prompt
        assert task.catalog.name == "golden"


def test_golden_task_to_dict_is_json_safe(golden_tasks: list[SqlTask]) -> None:
    payload = json.dumps(golden_tasks[0].to_dict(), default=str)
    assert "task_id" in payload


# ---------------------------------------------------------------------------
# SqlEnv lifecycle
# ---------------------------------------------------------------------------


def _scan_employees_task(catalog: GoldenCatalog) -> SqlTask:
    """Hand-rolled task: run the gold SQL and snapshot its rows."""
    rt = DialectRuntime(dialect=DIALECT, catalog=catalog)
    rt.setup()
    try:
        rows = rt.run("SELECT * FROM employees").exec_result.rows
    finally:
        rt.teardown()
    return SqlTask(
        meta=TaskMeta(task_id="t-scan", dialect=DIALECT, generator="test"),
        prompt="Return every employee.",
        gold_rows=rows,
        gold_sql="SELECT * FROM employees",
        catalog=catalog,
    )


def test_sql_env_step_and_done_match_path(runtime: DialectRuntime) -> None:
    task = _scan_employees_task(GoldenCatalog())
    env = SqlEnv(task=task, runtime=runtime, max_turns=2)
    obs0 = env.reset()
    assert obs0.task_id == "t-scan"
    assert obs0.dialect == DIALECT
    msgs = obs0.as_messages()
    assert msgs[0]["role"] == "system"

    step = env.step("SELECT * FROM employees")
    assert isinstance(step, StepResult)
    assert step.turn.matched
    assert step.done
    assert "OK" in step.observation
    assert env.done and not env.truncated

    # step() after done must raise (matches gym convention)
    with pytest.raises(RuntimeError):
        env.step("SELECT * FROM employees")


def test_sql_env_truncates_on_turn_budget(runtime: DialectRuntime) -> None:
    task = _scan_employees_task(GoldenCatalog())
    env = SqlEnv(task=task, runtime=runtime, max_turns=2)
    env.reset()
    s1 = env.step("totally not sql")
    assert not s1.done
    assert s1.turn.error_class == "parse"
    s2 = env.step("totally not sql")
    assert s2.done
    assert env.truncated


def test_sql_env_rejects_dialect_mismatch(runtime: DialectRuntime) -> None:
    bad_meta = TaskMeta(task_id="x", dialect="some_other_dialect", generator="test")
    task = SqlTask(
        meta=bad_meta, prompt="", gold_rows=[], gold_sql="", catalog=GoldenCatalog()
    )
    with pytest.raises(ValueError, match="does not match"):
        SqlEnv(task=task, runtime=runtime)


def test_sql_env_episode_result_carries_components(runtime: DialectRuntime) -> None:
    task = _scan_employees_task(GoldenCatalog())
    env = SqlEnv(task=task, runtime=runtime, max_turns=3)
    env.reset()
    env.step("SELECT * FROM employees")
    result = env.episode_result()
    assert isinstance(result, EpisodeResult)
    assert result.matched
    assert not result.truncated
    assert "correctness" in result.reward_components
    assert "turn_bonus" in result.reward_components
    # First-turn correct => full turn bonus contribution.
    assert result.reward_components["correctness"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def _turn(idx: int, *, matched: bool, error_class: str | None = None) -> Turn:
    return Turn(
        index=idx,
        sql="SELECT 1",
        exec_result=ExecResult(success=matched, rows=[], columns=[]),
        error_class=error_class,
        matched=matched,
    )


def test_compute_reward_first_turn_match_is_max() -> None:
    cmp_res = ComparisonResult(
        matches=True,
        exact_match=True,
        numeric_match=True,
        exact_distance=0.0,
        numeric_distance=0.0,
        f_score=1.0,
        reference_row_count=1,
        candidate_row_count=1,
        detail="match",
    )
    breakdown = compute_reward(
        transcript=[_turn(0, matched=True)],
        final_comparison=cmp_res,
        max_turns=3,
    )
    assert isinstance(breakdown, RewardBreakdown)
    assert breakdown.correctness == pytest.approx(5.0)
    assert breakdown.turn_bonus == pytest.approx(2.0)
    assert breakdown.error_shaping == 0.0
    assert breakdown.format_penalty == 0.0
    assert breakdown.total == pytest.approx(7.0)


def test_compute_reward_late_match_decays_turn_bonus() -> None:
    cmp_res = ComparisonResult(
        matches=True,
        exact_match=True,
        numeric_match=True,
        exact_distance=0.0,
        numeric_distance=0.0,
        f_score=1.0,
        reference_row_count=1,
        candidate_row_count=1,
        detail="match",
    )
    transcript = [
        _turn(0, matched=False, error_class="runtime"),
        _turn(1, matched=False, error_class="parse"),
        _turn(2, matched=True),
    ]
    breakdown = compute_reward(
        transcript=transcript,
        final_comparison=cmp_res,
        max_turns=3,
    )
    assert breakdown.correctness == pytest.approx(5.0)
    # 3rd of 3 turns => bonus is weight * (3 - 3 + 1) / 3 = 2.0/3
    assert breakdown.turn_bonus == pytest.approx(2.0 / 3)
    # runtime + parse penalties stack
    assert breakdown.error_shaping == pytest.approx(-0.5 + -1.0)


def test_compute_reward_never_parsed_triggers_format_penalty() -> None:
    # All-parse-error episode at the budget hits BOTH the format penalty
    # (never parsed) AND the terminal-invalid penalty (truncated with a
    # parse error on the last turn). They stack on top of the per-turn
    # parse shaping.
    breakdown = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class="parse"),
            _turn(1, matched=False, error_class="parse"),
        ],
        final_comparison=None,
        max_turns=2,
    )
    assert breakdown.correctness == 0.0
    assert breakdown.turn_bonus == 0.0
    assert breakdown.format_penalty == pytest.approx(-1.0)
    assert breakdown.error_shaping == pytest.approx(-2.0)
    assert breakdown.terminal_penalty == pytest.approx(-2.0)
    assert breakdown.total == pytest.approx(-2.0 + -1.0 + -2.0)


def test_compute_reward_partial_credit_capped() -> None:
    cmp_res = ComparisonResult(
        matches=False,
        exact_match=False,
        numeric_match=False,
        exact_distance=0.5,  # 50% jaccard distance
        numeric_distance=0.5,
        f_score=0.5,
        reference_row_count=4,
        candidate_row_count=2,
        detail="rows do not match",
    )
    cfg = RewardConfig(partial_credit=True)
    breakdown = compute_reward(
        transcript=[_turn(0, matched=False)],
        final_comparison=cmp_res,
        max_turns=3,
        config=cfg,
    )
    # 0.5 distance => 0.5 overlap => 5.0 * 0.4 * 0.5 = 1.0
    assert breakdown.correctness == pytest.approx(1.0)
    # No turn bonus on miss.
    assert breakdown.turn_bonus == 0.0


def test_compute_reward_partial_credit_can_be_disabled() -> None:
    cmp_res = ComparisonResult(
        matches=False,
        exact_match=False,
        numeric_match=False,
        exact_distance=0.5,
        numeric_distance=0.5,
        f_score=0.5,
        reference_row_count=4,
        candidate_row_count=2,
        detail="rows do not match",
    )
    breakdown = compute_reward(
        transcript=[_turn(0, matched=False)],
        final_comparison=cmp_res,
        max_turns=3,
        config=RewardConfig(partial_credit=False),
    )
    assert breakdown.correctness == 0.0


def test_compute_reward_terminal_invalid_fires_only_on_budget_with_parse_or_empty() -> None:
    # Truncated + last turn is parse error: terminal penalty fires.
    bd_parse = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class=None),     # parsed, wrong rows
            _turn(1, matched=False, error_class="parse"),  # regressed
        ],
        final_comparison=None,
        max_turns=2,
    )
    assert bd_parse.terminal_penalty == pytest.approx(-2.0)

    # Truncated + last turn is empty SQL: same terminal penalty.
    bd_empty = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class=None),
            _turn(1, matched=False, error_class="empty"),
        ],
        final_comparison=None,
        max_turns=2,
    )
    assert bd_empty.terminal_penalty == pytest.approx(-2.0)

    # Truncated + last turn parsed-but-wrong: NO terminal penalty
    # (the agent at least produced runnable SQL by the end).
    bd_unmatched = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class="parse"),
            _turn(1, matched=False, error_class=None),
        ],
        final_comparison=None,
        max_turns=2,
    )
    assert bd_unmatched.terminal_penalty == 0.0

    # Truncated + last turn was a runtime error: NO terminal penalty
    # (parseable SQL counts; runtime miss is a softer failure mode).
    bd_runtime = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class="runtime"),
            _turn(1, matched=False, error_class="runtime"),
        ],
        final_comparison=None,
        max_turns=2,
    )
    assert bd_runtime.terminal_penalty == 0.0


def test_compute_reward_terminal_invalid_does_not_fire_when_matched() -> None:
    cmp_match = ComparisonResult(
        matches=True, exact_match=True, numeric_match=True,
        exact_distance=0.0, numeric_distance=0.0, f_score=1.0,
        reference_row_count=1, candidate_row_count=1, detail="match",
    )
    bd = compute_reward(
        transcript=[_turn(0, matched=True)],
        final_comparison=cmp_match,
        max_turns=1,  # used the entire budget...
        config=RewardConfig(),
    )
    # ...but matched, so not truncated. No terminal penalty.
    assert bd.terminal_penalty == 0.0


# ---------------------------------------------------------------------------
# Discounted reward mode
# ---------------------------------------------------------------------------


def _match_cmp() -> ComparisonResult:
    return ComparisonResult(
        matches=True, exact_match=True, numeric_match=True,
        exact_distance=0.0, numeric_distance=0.0, f_score=1.0,
        reference_row_count=1, candidate_row_count=1, detail="match",
    )


def test_discounted_mode_first_turn_correct_is_unit() -> None:
    cfg = RewardConfig.discounted()
    bd = compute_reward(
        transcript=[_turn(0, matched=True)],
        final_comparison=_match_cmp(),
        max_turns=3,
        config=cfg,
    )
    # gamma^0 = 1.0; turn_bonus is folded into correctness in discounted mode.
    assert bd.correctness == pytest.approx(1.0)
    assert bd.turn_bonus == 0.0
    assert bd.error_shaping == 0.0
    assert bd.format_penalty == 0.0
    assert bd.terminal_penalty == 0.0
    assert bd.total == pytest.approx(1.0)


def test_discounted_mode_decays_with_turn_index() -> None:
    cfg = RewardConfig.discounted(discount_factor=0.9)
    transcript = [
        _turn(0, matched=False, error_class="parse"),
        _turn(1, matched=False, error_class="runtime"),
        _turn(2, matched=True),
    ]
    bd = compute_reward(
        transcript=transcript,
        final_comparison=_match_cmp(),
        max_turns=3,
        config=cfg,
    )
    # Matched on the 3rd turn => n=2 => 0.9^2 = 0.81.
    assert bd.correctness == pytest.approx(0.81)
    assert bd.turn_bonus == 0.0
    # Error shaping carries through identically to linear mode.
    assert bd.error_shaping == pytest.approx(-1.0 + -0.5)


def test_discounted_mode_no_partial_credit() -> None:
    cmp_partial = ComparisonResult(
        matches=False, exact_match=False, numeric_match=False,
        exact_distance=0.5, numeric_distance=0.5, f_score=0.5,
        reference_row_count=4, candidate_row_count=2,
        detail="rows do not match",
    )
    bd = compute_reward(
        transcript=[_turn(0, matched=False)],
        final_comparison=cmp_partial,
        max_turns=3,
        config=RewardConfig.discounted(),
    )
    # Discounted mode is binary: no partial credit even on a near-miss.
    assert bd.correctness == 0.0


def test_discounted_mode_custom_gamma() -> None:
    cfg = RewardConfig.discounted(discount_factor=0.5)
    bd = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class="runtime"),
            _turn(1, matched=False, error_class="runtime"),
            _turn(2, matched=True),
        ],
        final_comparison=_match_cmp(),
        max_turns=5,
        config=cfg,
    )
    # 0.5^2 = 0.25
    assert bd.correctness == pytest.approx(0.25)


def test_discounted_mode_terminal_penalty_still_applies() -> None:
    cfg = RewardConfig.discounted()
    bd = compute_reward(
        transcript=[
            _turn(0, matched=False, error_class="parse"),
            _turn(1, matched=False, error_class="parse"),
        ],
        final_comparison=None,
        max_turns=2,
        config=cfg,
    )
    # Even in discounted mode, a regressed-at-the-buzzer episode pays
    # the format + terminal-invalid penalties on top of the per-turn
    # parse shaping.
    assert bd.correctness == 0.0
    assert bd.format_penalty == pytest.approx(-1.0)
    assert bd.terminal_penalty == pytest.approx(-2.0)
    assert bd.error_shaping == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def test_run_episode_single_turn_match(runtime: DialectRuntime) -> None:
    task = _scan_employees_task(GoldenCatalog())
    env = SqlEnv(task=task, runtime=runtime, max_turns=2)
    policy = FixedSqlPolicy("SELECT * FROM employees")
    result = run_episode(env=env, policy=policy)
    assert result.episode.matched
    assert len(result.episode.turns) == 1
    # Messages: system + user + assistant + user-feedback.
    assert [m["role"] for m in result.messages] == ["system", "user", "assistant", "user"]


def test_run_episode_recovers_from_first_turn_error(runtime: DialectRuntime) -> None:
    task = _scan_employees_task(GoldenCatalog())
    env = SqlEnv(task=task, runtime=runtime, max_turns=3)
    policy = FixedSqlPolicy(
        sequence=["SELECT * FROM no_such_table", "SELECT * FROM employees"]
    )
    result = run_episode(env=env, policy=policy)
    assert result.episode.matched
    assert len(result.episode.turns) == 2
    assert result.episode.turns[0].error_class == "runtime"
    assert result.episode.turns[1].matched
    # Reward components reflect the bumpy path.
    rc = result.episode.reward_components
    assert rc["correctness"] == pytest.approx(5.0)
    # 2 of 3 turns used => bonus = 2.0 * (3 - 2 + 1) / 3 = 4/3
    assert rc["turn_bonus"] == pytest.approx(4.0 / 3)
    assert rc["error_shaping"] == pytest.approx(-0.5)
