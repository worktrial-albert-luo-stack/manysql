"""Result-set comparison metrics, ported from tinybirdco/llm-benchmark.

Mirrors `src/benchmark/result-validator.ts` from the upstream Node bench:

  * `jaccard_distance`: 1 - |intersect(A,B)| / |union(A,B)| over canonicalized rows.
  * `numeric_rmse_distance`: relative-error RMSE on aligned numeric cells.
  * `f_score_distance`: 1 - F1 over canonicalized rows.
  * `compare_results`: high-level wrapper that returns the same dict shape
    as the upstream `compareResults`, with an `exact` / `numeric` match flag.

Rows are canonicalized by sorting their values column-name-insensitively
so e.g. (`{x:1, y:'a'}`) and (`{a:'a', b:1}`) hash to the same bucket.
That matches the original behavior; it's a deliberate trade-off — column
name swaps go undetected, but the LLM is also not penalized for picking
different aliases than the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import sqrt
from typing import Any

Row = dict[str, Any]
Rows = list[Row]

EXACT_THRESHOLD = 0.05  # 5% Jaccard distance allowed for "exact" match
NUMERIC_THRESHOLD = 0.10  # 10% relative RMSE allowed for "numeric" match


@dataclass
class ComparisonResult:
    matches: bool
    exact_match: bool
    numeric_match: bool
    exact_distance: float
    numeric_distance: float
    f_score: float
    reference_row_count: int
    candidate_row_count: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "exact_match": self.exact_match,
            "numeric_match": self.numeric_match,
            "exact_distance": self.exact_distance,
            "numeric_distance": self.numeric_distance,
            "f_score": self.f_score,
            "reference_row_count": self.reference_row_count,
            "candidate_row_count": self.candidate_row_count,
            "detail": self.detail,
        }


def _canonical_row(row: Row) -> str:
    """Stable, column-name-insensitive, type-faithful id for a row.

    Used for the strict ``exact_match`` tier. Equal values of different
    types (``2015`` vs ``'2015'`` vs ``2015.0``) hash *differently* on
    purpose: a dialect that legitimately returns text-typed years should
    not be silently graded as identical to one that returns int years.
    Type/precision divergences show up as a failed exact_match but can
    still register as ``numeric_match`` via :func:`_canonical_row_loose`.
    """
    parts: list[str] = []
    for v in row.values():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"#{int(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"#{v}")
        else:
            parts.append(str(v))
    parts.sort()
    return "|".join(parts)


def _canonical_row_loose(row: Row) -> str:
    """Type-coercing variant of :func:`_canonical_row`.

    Used *only* by :func:`numeric_rmse_distance` to align rows across
    backends that disagree on storage types (e.g. SQLite's
    ``CAST(strftime(...) AS INTEGER)`` returning ``2015`` vs a synthetic
    dialect's ``SUBSTR(...)`` returning ``'2015'``). Numeric-looking
    strings are normalized to their numeric form so the row alignment
    works; the actual cell-by-cell distance is then computed on the
    raw values via :func:`_to_number`. This is intentionally NOT used
    for the strict ``exact_match`` tier so dialects retain freedom to
    distinguish text from numeric outputs.
    """
    parts: list[str] = []
    for v in row.values():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"#{int(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"#{float(v):.10g}")
        elif isinstance(v, str):
            stripped = v.strip()
            try:
                parts.append(f"#{float(stripped):.10g}")
            except (TypeError, ValueError):
                parts.append(v)
        else:
            parts.append(str(v))
    parts.sort()
    return "|".join(parts)


def _to_number(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def jaccard_distance(a: Rows, b: Rows) -> float:
    """0.0 = identical sets, 1.0 = disjoint."""
    set_a = {_canonical_row(r) for r in a}
    set_b = {_canonical_row(r) for r in b}
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def f_score_distance(a: Rows, b: Rows, beta: float = 1.0) -> float:
    """0.0 = perfect F-beta, 1.0 = no overlap."""
    set_a = {_canonical_row(r) for r in a}
    set_b = {_canonical_row(r) for r in b}
    inter = len(set_a & set_b)
    precision = (inter / len(set_a)) if set_a else 0.0
    recall = (inter / len(set_b)) if set_b else 0.0
    if precision == 0.0 and recall == 0.0:
        return 1.0
    f = (
        (1 + beta**2)
        * precision
        * recall
        / (beta**2 * precision + recall)
    )
    return 1.0 - f


def numeric_rmse_distance(a: Rows, b: Rows) -> float:
    """Relative RMSE between aligned numeric cells, clamped to [0, 1].

    Row alignment uses :func:`_canonical_row_loose` (type-coercing)
    so rows can pair up across backends whose storage types disagree
    -- e.g. ``year=2015`` (int) on SQLite vs ``year='2015'`` (text) on
    a SUBSTR-based dialect. The strict, type-faithful canonicalization
    is reserved for the ``exact_match`` tier.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    aa = sorted(a, key=_canonical_row_loose)
    bb = sorted(b, key=_canonical_row_loose)
    k = min(len(aa), len(bb))
    se = 0.0
    n = 0
    for i in range(k):
        vals_a = list(aa[i].values())
        vals_b = list(bb[i].values())
        m = min(len(vals_a), len(vals_b))
        for j in range(m):
            x = _to_number(vals_a[j])
            y = _to_number(vals_b[j])
            if math.isnan(x) or math.isnan(y):
                continue
            denom = (abs(x) + abs(y)) / 2 or 1.0
            delta = (x - y) / denom
            se += delta * delta
            n += 1
    if n == 0:
        return 1.0
    rmse = sqrt(se / n)
    return min(rmse, 1.0)


def compare_results(reference: Rows, candidate: Rows) -> ComparisonResult:
    """Top-level: return per-mode match flags and a summary detail string."""
    if reference is None or candidate is None:  # type: ignore[unreachable]
        return ComparisonResult(
            matches=False,
            exact_match=False,
            numeric_match=False,
            exact_distance=1.0,
            numeric_distance=1.0,
            f_score=0.0,
            reference_row_count=len(reference) if reference else 0,
            candidate_row_count=len(candidate) if candidate else 0,
            detail="missing data in results",
        )

    if len(reference) == 0 and len(candidate) == 0:
        return ComparisonResult(
            matches=True,
            exact_match=True,
            numeric_match=True,
            exact_distance=0.0,
            numeric_distance=0.0,
            f_score=1.0,
            reference_row_count=0,
            candidate_row_count=0,
            detail="both results empty",
        )

    exact = jaccard_distance(reference, candidate)
    numeric = numeric_rmse_distance(reference, candidate)
    f1 = 1.0 - f_score_distance(reference, candidate)

    is_exact = exact <= EXACT_THRESHOLD
    is_numeric = numeric <= NUMERIC_THRESHOLD

    if is_exact:
        detail = "match within exact threshold"
    elif is_numeric:
        detail = "match within numeric threshold"
    else:
        detail = "results do not match"

    return ComparisonResult(
        matches=is_exact or is_numeric,
        exact_match=is_exact,
        numeric_match=is_numeric,
        exact_distance=exact,
        numeric_distance=numeric,
        f_score=f1,
        reference_row_count=len(reference),
        candidate_row_count=len(candidate),
        detail=detail,
    )


__all__ = [
    "ComparisonResult",
    "compare_results",
    "f_score_distance",
    "jaccard_distance",
    "numeric_rmse_distance",
]
