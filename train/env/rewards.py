"""Reward functions for the SQL RL env.

The reward is a function of (correctness, turns_used). The agent is
encouraged to:

  * Produce a query that *parses* (avoid the parse-error penalty).
  * Produce a query that *runs* (avoid the runtime-error penalty).
  * Produce a query whose result rows match the gold rows
    (correctness reward, with partial credit for near-misses).
  * Do all of the above quickly (turn bonus that decays with turns
    used).

Two modes
---------

The ``mode`` knob on :class:`RewardConfig` switches between two
correctness/turn-efficiency formulations:

* ``"linear"`` (default) -- separate ``correct_weight`` (with optional
  partial credit) and ``turn_bonus_weight`` (linear decay of bonus over
  the turn budget). Interpretable, easy to tune component-by-component.

* ``"discounted"`` -- single multiplicative correctness:
  ``correctness = matches ? discount_factor**n : 0`` where ``n`` is
  the 0-based index of the turn the agent first matched on. Bounded
  in [0, 1], doesn't need re-tuning when ``max_turns`` changes, and
  matches the standard RL discounted-return semantics. Construct with
  :meth:`RewardConfig.discounted`. In this mode ``turn_bonus`` is
  always 0 (the discount IS the turn-efficiency signal).

Both modes share the same per-turn shaping penalties so a trainer can
A/B them by swapping the config without touching the rest of the
pipeline.

Design choices
--------------

* **Sparse correctness signal, dense shaping signals.** The terminal
  reward is dominated by row-match correctness; per-turn shaping comes
  from error-classification penalties (parse < runtime < unmatched).
* **Reward components, not a black-box scalar.** ``compute_reward``
  returns a :class:`RewardBreakdown` that decomposes the total. GRPO
  treats each reward function as its own logged metric, so trainers
  can plug each component in as a separate ``reward_fn``.
* **Terminal-failure penalty.** If the episode hits its turn budget
  AND the agent's final attempt was unparseable (parse / empty),
  apply an extra penalty on top of the per-turn shaping. The agent
  had every chance to recover and instead regressed at the buzzer;
  that's worse than failing the same way on turn 1.

Defaults are middle-of-the-road and meant to be swept over. The shape
of the breakdown (not the magnitudes) is the stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from eval.validator import ComparisonResult

if TYPE_CHECKING:
    from train.env.types import Turn


RewardMode = Literal["linear", "discounted"]


@dataclass
class RewardConfig:
    """Tunable weights for the SQL env reward.

    All weights are scalars; the total is a linear combination plus
    a constant offset. Setting a weight to 0.0 zeros out that
    component.
    """

    # Mode switch. ``linear`` = correct_weight + turn_bonus_weight (legacy,
    # interpretable). ``discounted`` = matched_correctness * discount**n
    # (clean RL semantics). See module docstring.
    mode: RewardMode = "linear"

    # ---------- linear-mode knobs ----------
    # Paid out exactly once at episode end on the FINAL attempt.
    # ``correct_weight`` is the ceiling for a fully-matching row set;
    # ``partial_credit`` enables graded credit for near-misses (capped
    # at 40% of the full reward; binary otherwise). Ignored in
    # ``discounted`` mode.
    correct_weight: float = 5.0
    partial_credit: bool = True
    # Paid out only if the final attempt matched. Linear decay in
    # turns_used: first-turn-correct yields the full bonus, last-turn-
    # correct yields ~ 1 / max_turns. Ignored in ``discounted`` mode.
    turn_bonus_weight: float = 2.0

    # ---------- discounted-mode knobs ----------
    # Multiplicative discount applied to a binary {0, 1} correctness
    # signal. n = 0-based index of the turn the agent first matched
    # on, so first-turn correct = discount_factor**0 = 1.0, second-
    # turn = discount_factor, etc. Ignored in ``linear`` mode.
    # 0.9 gives a gentle ramp (turn 5 still earns 0.66); 0.5 makes
    # multi-turn dramatically less rewarded.
    discount_factor: float = 0.9

    # ---------- per-turn shaping (both modes) ----------
    # Applied to EACH failed turn (not just the final one). Encourages
    # the model to fail soft (parse-correct but semantically wrong)
    # rather than fail hard (couldn't even produce parseable SQL).
    parse_error_penalty: float = -1.0
    runtime_error_penalty: float = -0.5
    empty_sql_penalty: float = -2.0
    unmatched_penalty: float = -0.25  # parsed + ran but rows wrong

    # ---------- terminal penalties (both modes) ----------
    # Format penalty: applied once if the agent never produced a
    # parseable query AT ANY turn -- including the degenerate case
    # where the agent didn't emit a <SQL> tag / tool call at all
    # (transcript length 0). Discourages catastrophic format collapse
    # AND closes the "refuse the task to avoid penalties" escape
    # hatch.
    never_parsed_penalty: float = -1.0
    # Terminal-invalid penalty: applied when the episode hits the
    # turn budget AND the final turn was a parse / empty error. The
    # agent had N chances and *regressed* to broken SQL on its last
    # try; that's strictly worse than running out of turns with
    # parseable-but-wrong SQL on the table. Stacks with the per-turn
    # parse penalty for the same turn.
    terminal_invalid_penalty: float = -2.0

    # -----------------------------------------------------------------
    # Presets
    # -----------------------------------------------------------------

    @classmethod
    def discounted(
        cls, *, discount_factor: float = 0.9, **kwargs: Any
    ) -> RewardConfig:
        """Build a ``mode="discounted"`` config with the given gamma.

        Extra keyword args are forwarded to ``__init__`` so callers can
        override e.g. shaping penalties without restating the mode flag.

        Examples::

            RewardConfig.discounted()                          # gamma=0.9
            RewardConfig.discounted(discount_factor=0.85)
            RewardConfig.discounted(parse_error_penalty=-0.5)  # softer shaping
        """
        return cls(mode="discounted", discount_factor=discount_factor, **kwargs)


@dataclass
class RewardBreakdown:
    """Per-component reward decomposition. ``total`` is the sum.

    Component shape is stable across modes so dashboards / W&B panels
    line up: ``turn_bonus`` is always present but is exactly 0.0 in
    discounted mode (the discount is folded into ``correctness``).
    """

    correctness: float
    turn_bonus: float
    error_shaping: float
    format_penalty: float
    terminal_penalty: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "turn_bonus": self.turn_bonus,
            "error_shaping": self.error_shaping,
            "format_penalty": self.format_penalty,
            "terminal_penalty": self.terminal_penalty,
            "total": self.total,
        }


def compute_reward(
    *,
    transcript: list[Turn],
    final_comparison: ComparisonResult | None,
    max_turns: int,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute the episode reward from a transcript + final comparison.

    Args:
        transcript: every turn the agent took, in order. Empty list
            means the agent never emitted a parseable SQL action (no
            <SQL> tag / no tool call). That is treated as a refusal:
            no correctness, no error shaping, and the full format
            penalty fires.
        final_comparison: row-match comparison against gold rows for
            the FINAL turn's exec result, or None if the final turn
            errored out (in which case correctness = 0).
        max_turns: the hard turn budget the env was configured with;
            used to scale the turn bonus and detect truncation.
        config: weights and toggles. Defaults to ``RewardConfig()``
            (linear mode).
    """
    cfg = config or RewardConfig()
    n_turns = len(transcript)
    matched = bool(final_comparison and final_comparison.matches)

    # ---- correctness + turn bonus (mode-dependent) ----
    if cfg.mode == "discounted":
        if matched:
            # n = 0-based index of the matching turn (always the last
            # turn in the transcript since the env stops on match).
            n = max(n_turns - 1, 0)
            correctness = cfg.discount_factor**n
        else:
            correctness = 0.0
        turn_bonus = 0.0  # subsumed into ``correctness`` by construction
    else:
        # Linear mode (legacy / default).
        correctness = (
            _linear_correctness(final_comparison, cfg) if final_comparison else 0.0
        )
        if matched and n_turns > 0:
            turn_bonus = (
                cfg.turn_bonus_weight
                * (max_turns - n_turns + 1)
                / max(max_turns, 1)
            )
        else:
            turn_bonus = 0.0

    # ---- per-turn error shaping (both modes) ----
    # Every failed turn contributes a small penalty weighted by how it
    # failed. The final turn is included so the agent can't game the
    # bonus by appending a known-good query after a string of bad ones
    # (the bad ones still cost it).
    error_shaping = 0.0
    parsed_at_least_once = False
    for turn in transcript:
        # A matched turn definitionally parsed + ran, so it counts as a
        # successful parse for the format-penalty detector even though
        # it contributes no error shaping.
        if turn.matched:
            parsed_at_least_once = True
            continue
        if turn.error_class == "parse":
            error_shaping += cfg.parse_error_penalty
        elif turn.error_class == "runtime":
            error_shaping += cfg.runtime_error_penalty
            parsed_at_least_once = True
        elif turn.error_class == "empty":
            error_shaping += cfg.empty_sql_penalty
        elif turn.error_class is None:
            error_shaping += cfg.unmatched_penalty
            parsed_at_least_once = True

    # Refusal accounting. An empty transcript means the agent never
    # produced an actionable SQL action at all (no <SQL> tag, no tool
    # call). Structurally that's a non-attempt, which we charge as
    # equivalent to a parse-error attempt for shaping purposes. Without
    # this, refusal would total -1.0 (just the format penalty below)
    # while a parse-error attempt totals -2.0 (parse shaping + format
    # penalty), and GRPO's group-relative advantage would systematically
    # prefer "say nothing" over "try and fail to parse" -- a degenerate
    # local optimum the model can collapse into early in training.
    if n_turns == 0:
        error_shaping += cfg.parse_error_penalty

    # ---- format penalty: catastrophic collapse ----
    # Fires whenever the agent never emitted a parseable SQL, including
    # the degenerate ``n_turns == 0`` case (no <SQL> tag / no tool call
    # at all). Stacks with the refusal-shaping penalty above so refusing
    # is never strictly better than honestly attempting and failing.
    format_penalty = 0.0
    if not parsed_at_least_once:
        format_penalty = cfg.never_parsed_penalty

    # ---- terminal-invalid penalty: regressed at the budget ----
    # Truncation = "agent ran out of turns without matching." Final
    # turn unparseable = "agent's last shot didn't even produce
    # runnable SQL." Both together => the agent had every chance and
    # blew the last one; punish more than a mid-episode parse error.
    truncated = (not matched) and (n_turns >= max_turns) and n_turns > 0
    terminal_penalty = 0.0
    if truncated:
        last = transcript[-1]
        if last.error_class in {"parse", "empty"}:
            terminal_penalty = cfg.terminal_invalid_penalty

    total = (
        correctness
        + turn_bonus
        + error_shaping
        + format_penalty
        + terminal_penalty
    )
    return RewardBreakdown(
        correctness=correctness,
        turn_bonus=turn_bonus,
        error_shaping=error_shaping,
        format_penalty=format_penalty,
        terminal_penalty=terminal_penalty,
        total=total,
    )


def _linear_correctness(
    comparison: ComparisonResult, cfg: RewardConfig
) -> float:
    """Map a ComparisonResult to a [0, correct_weight] correctness reward."""
    if comparison.matches:
        return cfg.correct_weight
    if not cfg.partial_credit:
        return 0.0
    # ``exact_distance`` is Jaccard distance over canonicalized rows;
    # 0 = identical, 1 = disjoint. Use (1 - distance) so non-zero
    # overlap yields a non-zero reward, scaled to the correctness budget.
    overlap = max(0.0, 1.0 - comparison.exact_distance)
    return cfg.correct_weight * 0.4 * overlap  # cap partial at 40% of the full reward


__all__ = [
    "RewardBreakdown",
    "RewardConfig",
    "RewardMode",
    "compute_reward",
]
