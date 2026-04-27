"""SqlEnv: the gym-style environment surface.

One ``SqlEnv`` instance corresponds to one episode against one
:class:`~train.env.tasks.SqlTask`. Lifecycle:

    env = SqlEnv(task=task, runtime=runtime, max_turns=3)
    obs = env.reset()
    while not env.done:
        sql = policy(obs)
        step = env.step(sql)
        obs = step.observation
    result = env.episode_result()  # transcript + reward + reward components

The env owns:

* The :class:`~train.env.tasks.SqlTask` (immutable - copied by reference).
* A :class:`~train.env.engine.DialectRuntime` (already setup) that runs
  candidate SQL against the dialect under test. The env does NOT own
  ``setup()`` lifecycle on the runtime, so the runtime can be reused
  across many episodes (one parser build, one catalog load, N tasks).
* A :class:`~train.env.rewards.RewardConfig` (immutable per env).

Termination conditions:

* ``done = True`` after ``env.step()`` if the candidate SQL's rows match
  the task's gold rows (within the validator's exact/numeric thresholds).
* ``done = True`` after the ``max_turns``-th step regardless.
* ``truncated = True`` iff ``done`` was triggered by the turn budget,
  not by a successful match.

The env produces an LLM-friendly ``observation`` string after each
``step``: a short success summary on match, a formatted error trace on
parse/runtime failure, or a "rows don't match yet" diff on
parsed-but-wrong. Callers building their own multi-turn prompts (e.g.
:class:`~train.env.rollout.LLMPolicy`) can override this and use the
underlying :class:`~train.env.types.StepResult.turn` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eval.validator import ComparisonResult, compare_results
from train.env.rewards import RewardConfig, compute_reward
from train.env.types import EpisodeResult, StepResult, Turn

if TYPE_CHECKING:
    from train.env.engine import DialectRuntime
    from train.env.tasks import SqlTask


@dataclass(frozen=True)
class InitialObservation:
    """Returned by ``reset()``: everything the agent needs to act once.

    Splits the prompt into ``system_prompt`` and ``user_message`` so the
    caller can format them to its model's chat template however it likes.
    """

    system_prompt: str
    user_message: str
    task_id: str
    dialect: str

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_message},
        ]


class SqlEnv:
    """Single-task, multi-turn SQL execution environment.

    Not thread-safe. Cheap to construct, but the underlying
    :class:`~train.env.engine.DialectRuntime` is the expensive resource;
    reuse one runtime across many envs.
    """

    def __init__(
        self,
        *,
        task: SqlTask,
        runtime: DialectRuntime,
        max_turns: int = 3,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if runtime.dialect != task.meta.dialect:
            # Soft mismatch warning: training pipelines that swap dialects
            # mid-episode would otherwise score against the wrong gold rows
            # silently.
            raise ValueError(
                f"runtime dialect {runtime.dialect!r} does not match task "
                f"dialect {task.meta.dialect!r}"
            )
        self.task = task
        self.runtime = runtime
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()
        self._turns: list[Turn] = []
        self._comparisons: list[ComparisonResult | None] = []
        self._done: bool = False
        self._truncated: bool = False
        self._reset_called: bool = False

    # -------- gym-ish surface --------

    def reset(self) -> InitialObservation:
        self._turns = []
        self._comparisons = []
        self._done = False
        self._truncated = False
        self._reset_called = True
        return InitialObservation(
            system_prompt=self.runtime.system_prompt(),
            user_message=self.task.prompt,
            task_id=self.task.meta.task_id,
            dialect=self.task.meta.dialect,
        )

    def step(self, sql: str) -> StepResult:
        """Run one candidate SQL through the dialect, score it, advance state."""
        if not self._reset_called:
            raise RuntimeError("SqlEnv.step() called before reset()")
        if self._done:
            raise RuntimeError("SqlEnv.step() called after episode finished")

        run = self.runtime.run(sql)
        comparison: ComparisonResult | None = None
        matched = False
        if run.exec_result.success:
            comparison = compare_results(self.task.gold_rows, run.exec_result.rows)
            matched = comparison.matches

        turn = Turn(
            index=len(self._turns),
            sql=sql,
            exec_result=run.exec_result,
            error_class=run.error_class,
            matched=matched,
        )
        self._turns.append(turn)
        self._comparisons.append(comparison)

        if matched:
            self._done = True
        elif len(self._turns) >= self.max_turns:
            self._done = True
            self._truncated = True

        observation = self._render_observation(turn, comparison)
        info: dict[str, Any] = {
            "turns_used": len(self._turns),
            "max_turns": self.max_turns,
            "truncated": self._truncated,
        }
        if comparison is not None:
            info["comparison"] = comparison.to_dict()
        return StepResult(turn=turn, observation=observation, done=self._done, info=info)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    # -------- episode finalization --------

    def episode_result(self) -> EpisodeResult:
        """Compute reward + assemble the EpisodeResult. Safe to call only after done."""
        if not self._done:
            raise RuntimeError("episode_result() called before episode finished")

        final_comparison = self._comparisons[-1] if self._comparisons else None
        breakdown = compute_reward(
            transcript=self._turns,
            final_comparison=final_comparison,
            max_turns=self.max_turns,
            config=self.reward_config,
        )
        matched = bool(final_comparison and final_comparison.matches)
        final_sql = self._turns[-1].sql if self._turns else None

        return EpisodeResult(
            task=self.task.meta,
            turns=list(self._turns),
            matched=matched,
            reward=breakdown.total,
            reward_components=breakdown.to_dict(),
            final_sql=final_sql,
            truncated=self._truncated,
        )

    # -------- internals --------

    def _render_observation(
        self, turn: Turn, comparison: ComparisonResult | None
    ) -> str:
        """Format a string for the LLM to read on its next turn.

        Mirrors the retry prompt ``eval/runner.py`` builds when the
        candidate SQL errors, plus a "your rows don't match" branch
        for the parsed-but-wrong case.
        """
        if turn.matched:
            n = len(turn.exec_result.rows)
            return f"OK. Returned {n} row{'s' if n != 1 else ''} matching the gold result."
        if not turn.exec_result.success:
            err = turn.exec_result.error or "(no error message)"
            label = {
                "parse": "PARSE ERROR",
                "runtime": "RUNTIME ERROR",
                "empty": "EMPTY SQL",
            }.get(turn.error_class or "", "ERROR")
            return (
                f"{label} on this query:\n{turn.sql}\n\n"
                f"Engine said:\n{err}\n\n"
                "Please rewrite the SQL so it parses and executes in the target dialect, "
                "and produces rows matching the question's gold result."
            )
        # Parsed + ran, but rows are wrong. Give the LLM a row-count hint
        # so it can spot off-by-one and missing-group bugs without leaking
        # the gold rows themselves.
        if comparison is None:  # defensive; shouldn't happen on success
            return "Query executed but result was not compared."
        return (
            f"Your query parsed and ran but the rows don't match the gold result.\n"
            f"  candidate rows: {comparison.candidate_row_count}\n"
            f"  gold rows:      {comparison.reference_row_count}\n"
            f"  exact_distance: {comparison.exact_distance:.3f} "
            f"(0.0 = identical sets)\n"
            f"  numeric_distance: {comparison.numeric_distance:.3f}\n"
            f"Try again."
        )


__all__ = ["InitialObservation", "SqlEnv"]
