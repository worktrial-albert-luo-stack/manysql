"""Lightweight dataclasses shared across the RL environment.

Every type here is JSON-serializable (via ``asdict()``) and side-effect
free. They define the protocol the rest of ``train.env`` speaks:

* ``TaskMeta``  - identity of the task being run (dialect, task id, etc).
* ``Turn``      - one (action, observation) pair the env logged.
* ``StepResult``- what ``SqlEnv.step()`` returns: observation + done flag.
* ``EpisodeResult`` - the full transcript + reward at the end of an episode.

The rationale for not pulling these straight from ``eval`` is twofold:
1. ``train.env`` should not depend on ``eval`` (layering).
2. RL trajectories carry per-turn metadata (turn index, terminal flag,
   reward components) that the eval ``Attempt`` shape doesn't have.

The execution-result shape we *do* reuse - ``eval.executors.base.ExecResult`` -
is a good fit and lives in a stable, dependency-light module so importing it
across the layer is fine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eval.executors.base import ExecResult


@dataclass(frozen=True)
class TaskMeta:
    """Identity + provenance for a single SQL task.

    A task is the (dialect, NL-question, expected-rows) tuple the agent
    is asked to solve in one episode. ``meta`` is free-form: task
    generators stash whatever they want here (golden id, category,
    generator name, seed, ...).
    """

    task_id: str
    dialect: str
    generator: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """One step of the agent's interaction with the env.

    ``index`` is 0-based. ``sql`` is exactly what the agent emitted
    (after extracting from any code fences; see
    ``eval.prompt.extract_sql``). ``exec_result`` is the dialect engine's
    response. ``error_class`` distinguishes parse vs runtime vs unknown
    errors so the reward function can weight them differently.
    """

    index: int
    sql: str
    exec_result: ExecResult
    error_class: str | None = None  # 'parse' | 'runtime' | 'empty' | None
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sql": self.sql,
            "exec_result": self.exec_result.to_dict(),
            "error_class": self.error_class,
            "matched": self.matched,
        }


@dataclass
class StepResult:
    """Return value of ``SqlEnv.step()``.

    ``observation`` is the textual feedback an LLM-style agent should
    see next turn; for runtime errors this is the formatted error trace.
    For success it's a short row-count summary. Callers driving the env
    with their own prompt format are free to ignore it.
    """

    turn: Turn
    observation: str
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn.to_dict(),
            "observation": self.observation,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class EpisodeResult:
    """Full record of one episode for offline analysis / RL training.

    ``reward_components`` decomposes the scalar so the trainer can log
    each (matches GRPO's "reward function returns a list" convention) -
    e.g. ``{"correctness": 0.8, "turn_bonus": 0.5, "errors": -0.1}``.
    """

    task: TaskMeta
    turns: list[Turn]
    matched: bool
    reward: float
    reward_components: dict[str, float]
    final_sql: str | None
    truncated: bool  # hit the turn budget without matching

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "turns": [t.to_dict() for t in self.turns],
            "matched": self.matched,
            "reward": self.reward,
            "reward_components": self.reward_components,
            "final_sql": self.final_sql,
            "truncated": self.truncated,
        }


__all__ = [
    "EpisodeResult",
    "StepResult",
    "TaskMeta",
    "Turn",
]
