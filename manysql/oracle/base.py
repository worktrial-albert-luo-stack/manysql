"""Oracle interface and capability metadata.

An Oracle takes an IR Plan + SemanticConfig + dataset catalog and either:
  - returns expected rows (`evaluate`), or
  - asserts a property holds (`assert_property`).

The verification harness selects the *strongest* applicable oracle per IR
feature, runs others opportunistically, and treats inter-oracle disagreement
as `needs_review`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from manysql.ir.plan import (
    Aggregate,
    Apply,
    Distinct,
    Filter,
    Join,
    Limit,
    Plan,
    Project,
    RecursiveCTE,
    Scan,
    SetOp,
    Sort,
    Window,
    WithCTE,
)
from manysql.spec.semantics import SemanticConfig


@dataclass(frozen=True)
class OracleCapability:
    """What an oracle can verify.

    `supported_nodes`: IR Plan classes (by name) the oracle can evaluate.
    `supported_features`: free-form feature tags (e.g. "correlated_subquery",
        "recursive_cte", "window_default_frame", "null_safe_eq").
    `unsupported_knobs`: SemanticConfig field names the oracle cannot honor.
        If a plan's evaluation depends on one of these knobs being non-default,
        the harness will skip this oracle.
    `confidence`: 0..1, used to break ties when multiple oracles apply.
        Higher means more trustworthy as primary.
    """

    name: str
    supported_nodes: frozenset[str]
    supported_features: frozenset[str] = field(default_factory=frozenset)
    unsupported_knobs: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 0.5


@dataclass
class OracleResult:
    """One oracle's evaluation result."""

    oracle: str
    rows: Optional[pl.DataFrame] = None  # None for property-only oracles
    property_passed: Optional[bool] = None
    error: Optional[str] = None
    notes: list[str] = field(default_factory=list)


class Oracle(ABC):
    """Base class for oracles."""

    @property
    @abstractmethod
    def capability(self) -> OracleCapability: ...

    def can_evaluate(self, plan: Plan, semantics: SemanticConfig) -> bool:
        """Default: check every plan node is in supported_nodes and no unsupported
        knobs are non-default."""
        cap = self.capability

        # Walk plan nodes
        if not _all_nodes_supported(plan, cap.supported_nodes):
            return False

        ref = SemanticConfig.reference()
        for knob in cap.unsupported_knobs:
            if getattr(semantics, knob) != getattr(ref, knob):
                return False
        return True

    @abstractmethod
    def evaluate(
        self,
        plan: Plan,
        semantics: SemanticConfig,
        catalog: dict[str, pl.DataFrame],
    ) -> OracleResult: ...


# ----- helpers -----


_NODE_TYPES: tuple[type[Plan], ...] = (
    Scan,
    Project,
    Filter,
    Join,
    Aggregate,
    Window,
    Sort,
    Limit,
    Distinct,
    SetOp,
    WithCTE,
    RecursiveCTE,
    Apply,
)


def _all_nodes_supported(plan: Plan, supported: frozenset[str]) -> bool:
    seen: set[str] = set()
    _collect_node_types(plan, seen)
    return seen.issubset(supported)


def _collect_node_types(plan: Plan, out: set[str]) -> None:
    out.add(type(plan).__name__)
    for child in plan.children():
        _collect_node_types(child, out)


def normalize_for_comparison(
    rows: pl.DataFrame,
    *,
    column_names: Optional[list[str]] = None,
    sort_keys: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Canonicalize a result frame for set/order-aware comparison.

    - Renames columns to `column_names` (if given) so two oracles with
      different default naming conventions can still match.
    - Sorts rows by `sort_keys` (or all columns if None) so unordered queries
      compare set-wise. Order-sensitive plans (Sort/Limit) skip this and
      compare position-wise instead.
    """
    df = rows
    if column_names is not None:
        if len(column_names) != df.width:
            raise ValueError(
                f"normalize_for_comparison: expected {len(column_names)} cols, got {df.width}"
            )
        df = df.rename(dict(zip(df.columns, column_names, strict=True)))
    if sort_keys is None:
        sort_keys = list(df.columns)
    if sort_keys:
        df = df.sort(by=sort_keys, nulls_last=True)
    return df


def frames_equal(a: pl.DataFrame, b: pl.DataFrame, *, ordered: bool = False) -> tuple[bool, str]:
    """Compare two result frames; returns (equal, reason)."""
    if a.columns != b.columns:
        return False, f"column mismatch: {a.columns} vs {b.columns}"
    if a.height != b.height:
        return False, f"row-count mismatch: {a.height} vs {b.height}"
    if not ordered:
        a = a.sort(by=a.columns, nulls_last=True)
        b = b.sort(by=b.columns, nulls_last=True)
    try:
        if a.equals(b):
            return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"comparison error: {exc}"
    # Find first differing row
    for i in range(a.height):
        if a.row(i) != b.row(i):
            return False, f"row {i}: {a.row(i)} vs {b.row(i)}"
    return False, "frames not equal but no row diff found"


def is_order_sensitive(plan: Plan) -> bool:
    """A plan whose top-most operator is order-sensitive (Sort or Limit)
    must be compared position-wise, not as a set."""
    return isinstance(plan, (Sort, Limit))


__all__ = [
    "Oracle",
    "OracleCapability",
    "OracleResult",
    "normalize_for_comparison",
    "frames_equal",
    "is_order_sensitive",
]
