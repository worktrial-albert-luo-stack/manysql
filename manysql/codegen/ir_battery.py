"""IR-equivalence battery: validate that a generated dialect's lowering
produces the same IR plan as the reference for the same logical query.

For each canonical SQL we hold:
    (label, reference_surface_sql, dialect_surface_sql)

The reference SQL is parsed + lowered using the reference dialect to obtain
`ref_plan`. The dialect SQL is parsed + lowered using the candidate
grammar/lowering to obtain `dialect_plan`. The two plans should be equal
(IR dataclasses are frozen with structural equality).

Why "should": for purely surface dialects (renamed keywords, same shape),
both grammars expose identical rule trees and the lowering is reused
verbatim. For structural dialects the LLM lane has to bridge the gap, and
this battery is what tells it whether it succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from types import ModuleType
from typing import Optional

from lark import Lark
from lark.exceptions import LarkError

from manysql.codegen.parse_battery import _REFERENCE_SQL, apply_surface
from manysql.ir.plan import ColumnSchema, Plan
from manysql.ir.printer import render_plan
from manysql.spec.dialect import DialectSpec, LimitSyntax
from manysql.spec.semantics import SemanticConfig
from manysql.storage import CATALOG, schema_of


# Limit syntaxes whose surface form cannot encode an OFFSET. The surface
# rewriter (`_format_limit_clause`) silently drops the offset for these,
# so a battery item like `LIMIT 5 OFFSET 10` cannot possibly round-trip
# back to an offset=10 plan. We skip those items rather than punish the
# lowering for an impossible reconstruction.
_OFFSET_CAPABLE_LIMIT_SYNTAXES: frozenset[LimitSyntax] = frozenset(
    {LimitSyntax.LIMIT_OFFSET, LimitSyntax.OFFSET_FETCH}
)
_OFFSET_DEPENDENT_LABELS: frozenset[str] = frozenset({"limit_offset"})


Catalog = dict[str, tuple[ColumnSchema, ...]]


@dataclass(frozen=True)
class IRBatteryItem:
    label: str
    ref_sql: str
    dialect_sql: str


@dataclass(frozen=True)
class IRDivergence:
    label: str
    ref_sql: str
    dialect_sql: str
    ref_plan: Optional[str]
    dialect_plan: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class IREquivalenceReport:
    items: list[IRBatteryItem]
    divergences: list[IRDivergence]

    @property
    def ok(self) -> bool:
        return not self.divergences

    def summary(self) -> str:
        if self.ok:
            return f"ir battery: {len(self.items)} / {len(self.items)} OK"
        return (
            f"ir battery: {len(self.items) - len(self.divergences)} / "
            f"{len(self.items)} OK; divergences: "
            + ", ".join(d.label for d in self.divergences)
        )


def build_ir_battery(spec: DialectSpec) -> list[IRBatteryItem]:
    """Pair each canonical reference SQL with its surface-rewritten form.

    Battery items that depend on a surface feature the spec's syntax
    cannot encode (currently: ``OFFSET`` for ``HEAD_N`` / ``SAMPLE_N`` /
    ``TOP_N``) are skipped so the lowering isn't asked to reconstruct
    information that was discarded by the rewriter.
    """
    skip_labels = _skipped_labels(spec)
    items: list[IRBatteryItem] = []
    for label, ref_sql in _REFERENCE_SQL:
        if label in skip_labels:
            continue
        items.append(
            IRBatteryItem(
                label=label,
                ref_sql=ref_sql,
                dialect_sql=apply_surface(ref_sql, spec.surface),
            )
        )
    return items


def _skipped_labels(spec: DialectSpec) -> frozenset[str]:
    skipped: set[str] = set()
    if spec.surface.limit_syntax not in _OFFSET_CAPABLE_LIMIT_SYNTAXES:
        skipped.update(_OFFSET_DEPENDENT_LABELS)
    return frozenset(skipped)


def default_schemas() -> Catalog:
    """Return the catalog used for the parse/IR batteries.

    The IR battery doesn't execute, so we just need column-name + type
    metadata, not actual data.
    """
    return {name: schema_of(name) for name in CATALOG}


def validate_lowering(
    *,
    grammar_text: str,
    lowering_module: ModuleType,
    semantics: SemanticConfig,
    items: list[IRBatteryItem],
    schemas: Optional[Catalog] = None,
) -> IREquivalenceReport:
    """Compare lowering of `dialect_sql` against the reference lowering of `ref_sql`.

    Returns a structured report; never raises (per-item failures are captured).
    """
    schemas = schemas or default_schemas()
    try:
        dialect_parser = Lark(grammar_text, start="start", parser="earley")
    except LarkError as exc:
        return IREquivalenceReport(
            items=items,
            divergences=[
                IRDivergence(
                    label=item.label,
                    ref_sql=item.ref_sql,
                    dialect_sql=item.dialect_sql,
                    ref_plan=None,
                    dialect_plan=None,
                    error=f"grammar build failed: {exc}",
                )
                for item in items
            ],
        )

    ref_parser, ref_lower = _reference_pipeline()

    divergences: list[IRDivergence] = []
    for item in items:
        try:
            ref_tree = ref_parser.parse(item.ref_sql)
            ref_plan = ref_lower(ref_tree, semantics, schemas)
        except Exception as exc:
            divergences.append(
                IRDivergence(
                    label=item.label,
                    ref_sql=item.ref_sql,
                    dialect_sql=item.dialect_sql,
                    ref_plan=None,
                    dialect_plan=None,
                    error=f"reference lowering failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        try:
            dialect_tree = dialect_parser.parse(item.dialect_sql)
            dialect_plan = lowering_module.lower(dialect_tree, semantics, schemas)
        except Exception as exc:
            divergences.append(
                IRDivergence(
                    label=item.label,
                    ref_sql=item.ref_sql,
                    dialect_sql=item.dialect_sql,
                    ref_plan=render_plan(ref_plan),
                    dialect_plan=None,
                    error=f"dialect lowering failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        if not _plans_equal(ref_plan, dialect_plan):
            divergences.append(
                IRDivergence(
                    label=item.label,
                    ref_sql=item.ref_sql,
                    dialect_sql=item.dialect_sql,
                    ref_plan=render_plan(ref_plan),
                    dialect_plan=render_plan(dialect_plan),
                    error="plan mismatch",
                )
            )
    return IREquivalenceReport(items=items, divergences=divergences)


def _reference_pipeline() -> tuple[Lark, callable]:
    """Build a (parser, lower_fn) pair for the reference dialect."""
    grammar = resources.read_text(
        "manysql.dialects._reference", "grammar.lark", encoding="utf-8"
    )
    parser = Lark(grammar, start="start", parser="earley")
    from manysql.dialects._reference import lowering as ref_lowering

    return parser, ref_lowering.lower


def _plans_equal(a: Plan, b: Plan) -> bool:
    """Frozen dataclasses compare structurally, but make this explicit so
    future relaxations (e.g. semantic equivalence under reordering) live in
    one place."""
    return a == b


__all__ = [
    "Catalog",
    "IRBatteryItem",
    "IRDivergence",
    "IREquivalenceReport",
    "build_ir_battery",
    "default_schemas",
    "validate_lowering",
]
