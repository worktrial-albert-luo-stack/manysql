"""Episode rollouts: drive a policy through a SqlEnv.

A :class:`Policy` is anything with a ``__call__(prompt) -> str`` shape.
The rollout helper :func:`run_episode` resets the env, hands the
initial messages to the policy, asks it for SQL, steps the env, and
loops until ``done``. Two policies ship in this module:

* :class:`FixedSqlPolicy` - always returns the same canned SQL string.
  Useful in tests + for sanity-checking the env's gold rows by replaying
  the reference SQL.
* :class:`LLMPolicy` - wraps an ``eval.llm.LLMClient`` and grows the
  message stack across turns so the model sees its own history + the
  env's per-turn observations (parse / runtime errors, "rows don't
  match" hints).

Trainers (GRPO et al.) generally do not call :class:`LLMPolicy`
directly: they batch generations and need explicit control over the
message stack so they can score per-completion. They DO use
:class:`SqlEnv` directly + the per-turn ``observation`` text. See
``train/env/README.md`` for the GRPO integration sketch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from eval.prompt import extract_sql
from train.env.sql_env import InitialObservation, SqlEnv

if TYPE_CHECKING:
    from eval.llm import LLMClient
    from train.env.types import EpisodeResult


# ---------------------------------------------------------------------------
# Policy ABC
# ---------------------------------------------------------------------------


class Policy(ABC):
    """ABC for SQL-generating policies.

    The contract is intentionally minimal: a policy receives the
    accumulated chat history (system + user messages, plus any prior
    assistant turns the policy chose to track) and returns the next
    SQL string. The env strips fences via :func:`extract_sql` before
    parsing, so the policy is free to wrap its output in markdown if
    it wants.
    """

    name: str = "abstract"

    @abstractmethod
    def act(self, messages: list[dict[str, str]]) -> str:
        """Produce the next SQL given the conversation so far."""

    def reset(self) -> None:  # noqa: B027 - intentional no-op default
        """Optional per-episode reset hook (e.g. to clear scratch state)."""


# ---------------------------------------------------------------------------
# FixedSqlPolicy: returns canned SQL
# ---------------------------------------------------------------------------


class FixedSqlPolicy(Policy):
    """Always returns the same SQL.

    If ``sequence`` is provided, returns each entry in order then
    repeats the last one. Useful for testing the multi-turn loop:
    pass [bad_sql, good_sql] to verify retry behavior.
    """

    name = "fixed"

    def __init__(self, sql: str | None = None, *, sequence: list[str] | None = None) -> None:
        if sql is None and not sequence:
            raise ValueError("FixedSqlPolicy needs sql= or sequence=")
        self._sequence: list[str] = list(sequence) if sequence else [sql or ""]
        self._idx: int = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self, messages: list[dict[str, str]]) -> str:
        del messages  # FixedSqlPolicy ignores the conversation by design
        if self._idx < len(self._sequence):
            out = self._sequence[self._idx]
            self._idx += 1
        else:
            out = self._sequence[-1]
        return out


# ---------------------------------------------------------------------------
# LLMPolicy: wraps an LLMClient
# ---------------------------------------------------------------------------


class LLMPolicy(Policy):
    """Adapter that turns an ``eval.llm.LLMClient`` into a :class:`Policy`.

    Sends the full message stack (which the rollout helper grows over
    turns) to ``client.chat(messages=...)`` each turn. Keeps zero
    state of its own; the caller owns the conversation.
    """

    name = "llm"

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def act(self, messages: list[dict[str, str]]) -> str:
        # Send via messages= so we get the full context every turn.
        # The client's `system`/`user` shortcut is just a sugar for the
        # 1-shot case; here we need the assistant<->user turn-taking.
        # `system`/`user` are both required by the dataclass-style API
        # but ignored when ``messages`` is provided.
        resp = self.client.chat(system="", user="", messages=messages)
        return extract_sql(resp.text)


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Convenience wrapper around (episode_result, message_stack).

    The message stack is the full chat transcript -- useful for
    rendering training data (instruction-tuning targets, GRPO
    completions) and for human inspection in eval dashboards.
    """

    episode: EpisodeResult
    messages: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "episode": self.episode.to_dict(),
            "messages": list(self.messages),
        }


def run_episode(
    *,
    env: SqlEnv,
    policy: Policy,
    extract: bool = True,
) -> RolloutResult:
    """Run one episode end-to-end.

    Maintains the full chat history (system + initial user + per-turn
    assistant SQL + per-turn user observation) and feeds it back to
    the policy on every step. Stops when the env reports ``done``.

    Args:
        env: a fresh :class:`SqlEnv`. ``reset()`` is called here.
        policy: the SQL-generating policy. ``reset()`` is called here.
        extract: if True, run :func:`extract_sql` on each policy
            output to strip code fences. ``LLMPolicy`` already does
            this; the flag exists for raw ``FixedSqlPolicy`` use that
            wants to test malformed input.
    """
    policy.reset()
    obs0: InitialObservation = env.reset()
    messages: list[dict[str, str]] = obs0.as_messages()

    while not env.done:
        raw = policy.act(messages)
        sql = extract_sql(raw) if extract else raw

        # Persist what the agent emitted (post-extraction, since that's
        # what the env actually evaluated). For raw transcripts the
        # caller can inspect ``messages`` itself.
        messages.append({"role": "assistant", "content": sql})

        step = env.step(sql)
        messages.append({"role": "user", "content": step.observation})

    return RolloutResult(episode=env.episode_result(), messages=messages)


__all__ = [
    "FixedSqlPolicy",
    "LLMPolicy",
    "Policy",
    "RolloutResult",
    "run_episode",
]
