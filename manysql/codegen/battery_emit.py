"""Emit human-readable ``examples.sql`` and machine-readable ``battery.json``.

These artifacts capture the *reskinned* parse and IR-equivalence batteries —
the same canonical reference SQL queries, rewritten into the dialect's surface
form — and ship them alongside the rest of the dialect package.

Why persist them?
- Documentation: ``examples.sql`` shows by eye what the dialect's surface
  actually looks like (e.g. ``PICK . OUT_OF employees`` for a swap dialect).
- Regression: the saved battery travels with the dialect, so anyone editing
  ``grammar.lark`` or ``lowering.py`` later can re-validate without
  recomputing from ``_REFERENCE_SQL``.
- Triage: ``battery.json`` records the final pass/fail summary so you can
  tell at a glance whether the package was written from a clean validation
  or with known divergences (e.g. a structural-change spec that needed LLM
  refinement and still has open items).

The emitters are pure functions of inputs the codegen pipeline already has
on hand — no recomputation, just plumbing.
"""

from __future__ import annotations

import json

from manysql.codegen.ir_battery import IRBatteryItem, IREquivalenceReport
from manysql.codegen.parse_battery import BatteryItem, ValidationReport
from manysql.spec.dialect import DialectSpec


def emit_battery_json(
    *,
    parse_items: list[BatteryItem],
    parse_report: ValidationReport,
    ir_items: list[IRBatteryItem],
    ir_report: IREquivalenceReport,
) -> str:
    """Return JSON text for the dialect's ``battery.json``.

    Schema is intentionally flat and human-readable. Failures embed enough
    context (label + offending SQL + error string) to triage without needing
    to re-run codegen, but we deliberately omit full plan dumps — those live
    in ``metadata.json``'s retry log if anywhere.
    """
    payload = {
        "parse": {
            "items": [
                {"label": item.label, "source": item.source}
                for item in parse_items
            ],
            "validation": {
                "ok": parse_report.ok,
                "summary": parse_report.summary(),
                "failures": [
                    {
                        "label": f.label,
                        "source": f.source,
                        "error": f.error,
                    }
                    for f in parse_report.failures
                ],
            },
        },
        "ir_equivalence": {
            "items": [
                {
                    "label": item.label,
                    "ref_sql": item.ref_sql,
                    "dialect_sql": item.dialect_sql,
                }
                for item in ir_items
            ],
            "validation": {
                "ok": ir_report.ok,
                "summary": ir_report.summary(),
                "divergences": [
                    {
                        "label": d.label,
                        "ref_sql": d.ref_sql,
                        "dialect_sql": d.dialect_sql,
                        "error": d.error,
                    }
                    for d in ir_report.divergences
                ],
            },
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def emit_examples_sql(
    *,
    spec: DialectSpec,
    parse_items: list[BatteryItem],
) -> str:
    """Return human-readable SQL text for the dialect's ``examples.sql``.

    One labeled section per canonical query, separated by blank lines.
    Statements are emitted exactly as they were validated by the parse
    battery (so any spec-mandated terminator is already baked in).
    """
    header = (
        f"-- manysql-codegen examples for dialect: {spec.name}\n"
        "-- Hand-curated canonical SQL queries rewritten into this dialect's surface.\n"
        "-- These are the same items used by the parse and IR-equivalence batteries.\n"
        f"-- Re-generate with: manysql-codegen {spec.name} --overwrite\n"
        "\n"
    )
    sections = [f"-- {item.label}\n{item.source}\n" for item in parse_items]
    return header + "\n".join(sections)


__all__ = [
    "emit_battery_json",
    "emit_examples_sql",
]
