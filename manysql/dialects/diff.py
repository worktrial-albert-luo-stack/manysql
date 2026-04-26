"""Diff a generated dialect's reskinned battery against the reference.

The "reskinned battery" is the canonical reference SQL corpus rewritten into
the dialect's surface form via ``apply_surface``. For dialects with surface
divergences (renamed keywords, alternate LIMIT/CAST/CASE syntax, etc.) some
items will differ from the reference; for purely-semantic dialects every
item will be identical.

This module is the data layer for the ``manysql-dialect diff`` CLI:

- ``compute_battery_diff`` is the pure-data side. It pulls each item's
  reference SQL and dialect SQL out of ``battery.json`` (preferred) or
  recomputes via ``apply_surface`` if the package predates ``battery.json``.
- ``render_battery_diff_table`` and ``render_battery_diff_unified`` are the
  presentation side; both consume a ``BatteryDiff``.

Keeping the two halves separate means tests can assert on the data without
parsing terminal output.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.table import Table

from manysql.codegen.parse_battery import _REFERENCE_SQL, apply_surface
from manysql.spec.dialect import SurfaceSpec


@dataclass(frozen=True)
class BatteryDiffItem:
    label: str
    reference_sql: str
    dialect_sql: str

    @property
    def changed(self) -> bool:
        return self.reference_sql != self.dialect_sql


@dataclass(frozen=True)
class BatteryDiff:
    dialect_name: str
    items: list[BatteryDiffItem]
    source: str  # "battery.json" | "spec.json (recomputed)"

    @property
    def changed_count(self) -> int:
        return sum(1 for it in self.items if it.changed)

    @property
    def total_count(self) -> int:
        return len(self.items)


def compute_battery_diff(
    dialect_path: Path,
    *,
    dialect_name: Optional[str] = None,
) -> BatteryDiff:
    """Pair each reference SQL item with the dialect's reskinned version.

    Prefers ``battery.json`` (the as-validated form) and falls back to
    recomputing from ``spec.json`` for older packages.
    """
    name = dialect_name or dialect_path.name
    battery_path = dialect_path / "battery.json"
    if battery_path.exists():
        battery = json.loads(battery_path.read_text())
        ir_items = battery.get("ir_equivalence", {}).get("items", [])
        if ir_items:
            return BatteryDiff(
                dialect_name=name,
                items=[
                    BatteryDiffItem(
                        label=item["label"],
                        reference_sql=item["ref_sql"],
                        dialect_sql=item["dialect_sql"],
                    )
                    for item in ir_items
                ],
                source="battery.json",
            )

    spec_path = dialect_path / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Dialect package at {dialect_path} has neither a usable "
            "battery.json nor a spec.json to recompute from."
        )
    spec_data = json.loads(spec_path.read_text())
    surface_data = spec_data.get("surface")
    if surface_data is None:
        raise ValueError(
            f"spec.json at {spec_path} is missing a 'surface' field; "
            "cannot recompute the battery."
        )
    surface = SurfaceSpec(**surface_data)
    return BatteryDiff(
        dialect_name=name,
        items=[
            BatteryDiffItem(
                label=label,
                reference_sql=ref_sql,
                dialect_sql=apply_surface(ref_sql, surface),
            )
            for label, ref_sql in _REFERENCE_SQL
        ],
        source="spec.json (recomputed)",
    )


def render_battery_diff_table(
    diff: BatteryDiff,
    *,
    only_changed: bool = False,
) -> Table:
    """Render the diff as a Rich side-by-side table.

    Identical rows are dimmed; reskinned rows show the dialect SQL in yellow
    so changes pop visually. Long SQL strings wrap (``overflow='fold'``) so
    nothing is truncated even on narrow terminals.
    """
    title = (
        f"battery diff: {diff.dialect_name} vs reference "
        f"({diff.changed_count}/{diff.total_count} reskinned, "
        f"source: {diff.source})"
    )
    table = Table(title=title, show_lines=True)
    table.add_column("label", style="dim", no_wrap=True)
    table.add_column("reference SQL", style="cyan", overflow="fold")
    table.add_column("dialect SQL", overflow="fold")
    for item in diff.items:
        if only_changed and not item.changed:
            continue
        if item.changed:
            dialect_cell = f"[yellow]{item.dialect_sql}[/yellow]"
        else:
            dialect_cell = f"[dim]{item.dialect_sql}[/dim]"
        table.add_row(item.label, item.reference_sql, dialect_cell)
    return table


def render_battery_diff_unified(
    diff: BatteryDiff,
    *,
    only_changed: bool = False,
) -> str:
    """Render the diff as plain-text unified diff blocks, one per item.

    Useful for piping into ``less``, attaching to bug reports, or stashing
    in CI logs where Rich color codes aren't welcome.
    """
    blocks: list[str] = []
    for item in diff.items:
        if only_changed and not item.changed:
            continue
        if not item.changed:
            blocks.append(
                f"=== {item.label} (unchanged) ===\n  {item.reference_sql}\n"
            )
            continue
        diff_lines = list(
            difflib.unified_diff(
                [item.reference_sql],
                [item.dialect_sql],
                fromfile=f"reference/{item.label}",
                tofile=f"{diff.dialect_name}/{item.label}",
                lineterm="",
            )
        )
        blocks.append("=== " + item.label + " ===\n" + "\n".join(diff_lines))
    summary = (
        f"\n--- {diff.changed_count}/{diff.total_count} items reskinned "
        f"(source: {diff.source}) ---"
    )
    return "\n\n".join(blocks) + summary


__all__ = [
    "BatteryDiff",
    "BatteryDiffItem",
    "compute_battery_diff",
    "render_battery_diff_table",
    "render_battery_diff_unified",
]
