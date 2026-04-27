"""CLI: ``python -m manysql.dialects`` / ``manysql-dialect``.

Inspect already-generated dialect packages. Right now the only subcommand is
``diff``, which shows how a dialect's reskinned battery (parse + IR
batteries rewritten via ``apply_surface``) compares to the canonical
reference SQL. Future subcommands (``list``, ``show``, ``lint``, ...) will
slot in alongside it.

Examples
--------

    # side-by-side Rich table
    manysql-dialect diff aggressive_alien

    # plain unified diff per item, only items that actually changed
    manysql-dialect diff aggressive_alien --unified --changed-only

    # against a custom dialects root (e.g. a temp registry from a smoke test)
    manysql-dialect diff aggressive_alien --dialects-dir /tmp/manysql-demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from manysql.dialects.diff import (
    compute_battery_diff,
    render_battery_diff_table,
    render_battery_diff_unified,
)
from manysql.dialects.registry import DIALECTS_DIR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manysql-dialect",
        description="Inspect generated manysql dialect packages.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    diff = sub.add_parser(
        "diff",
        help="Show the diff between a dialect's reskinned battery and the reference.",
        description=(
            "Side-by-side comparison of each canonical reference SQL query and "
            "its rewritten form in the dialect's surface. Defaults to a Rich "
            "table; pass --unified for a plain-text unified diff."
        ),
    )
    diff.add_argument(
        "name",
        help="Dialect name (directory under --dialects-dir).",
    )
    diff.add_argument(
        "--dialects-dir",
        default=None,
        help="Override the dialects root (default: manysql/dialects/).",
    )
    diff.add_argument(
        "--unified",
        action="store_true",
        help="Output plain-text unified diff blocks instead of a Rich table.",
    )
    diff.add_argument(
        "--changed-only",
        action="store_true",
        help="Hide items that are identical to the reference.",
    )
    return parser


def _cmd_diff(args: argparse.Namespace) -> int:
    console = Console()
    root = Path(args.dialects_dir) if args.dialects_dir else DIALECTS_DIR
    dialect_path = root / args.name
    if not dialect_path.is_dir():
        console.print(
            f"[red]error[/red]: no dialect package at {dialect_path}",
        )
        console.print(
            "[yellow]hint:[/yellow] run "
            f"[cyan]manysql-codegen gen {args.name}[/cyan] first, or pass "
            "--dialects-dir to point at the right registry root.",
        )
        return 1
    try:
        diff = compute_battery_diff(dialect_path, dialect_name=args.name)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error[/red]: {exc}")
        return 1

    if args.unified:
        # Print as plain text so redirected output stays clean.
        print(render_battery_diff_unified(diff, only_changed=args.changed_only))
    else:
        console.print(
            render_battery_diff_table(diff, only_changed=args.changed_only)
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "diff":
        return _cmd_diff(args)
    return 2  # pragma: no cover - argparse rejects unknown subcommands first


if __name__ == "__main__":
    sys.exit(main())
