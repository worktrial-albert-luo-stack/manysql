"""CLI: ``python -m manysql.codegen`` / ``manysql-codegen``.

Drives `manysql.codegen.pipeline.write_dialect_package` to materialize a
new dialect package under `manysql/dialects/<name>/`. Designed to be the
ergonomic counterpart to ``manysql-eval``: generate a dialect, then eval
against it.

Examples
--------

    # list bundled spec examples
    manysql-codegen --list

    # generate via the deterministic emitter (no LLM, near-instant)
    manysql-codegen mild_postgres_ish

    # ALWAYS run at least one LLM refinement pass on top of the deterministic
    # baseline (LLM output is rolled back if it regresses the battery).
    # Uses OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY from env.
    manysql-codegen aggressive_alien --use-llm

    # arbitrary spec from an importable Python attribute
    manysql-codegen mypkg.specs:MY_SPEC --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from manysql.codegen.pipeline import (
    BatteryError,
    PackageWriteResult,
    write_dialect_package,
)
from manysql.dialects.registry import DIALECTS_DIR, DialectRegistry, Lifecycle
from manysql.spec.dialect import DialectSpec

# Bundled example specs ship under manysql.spec.examples. We keep this
# table small and explicit so `--list` is helpful out of the box.
_BUNDLED_EXAMPLES: dict[str, str] = {
    "mild_postgres_ish": "manysql.spec.examples.mild_postgres_ish:MILD_POSTGRES_ISH",
    "moderate_keyword_swap": "manysql.spec.examples.moderate_keyword_swap:MODERATE_KEYWORD_SWAP",
    "aggressive_alien": "manysql.spec.examples.aggressive_alien:AGGRESSIVE_ALIEN",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manysql-codegen",
        description=(
            "Materialize a manysql dialect package from a DialectSpec. "
            "Run `manysql-codegen --list` to see bundled examples."
        ),
    )
    p.add_argument(
        "spec",
        nargs="?",
        help="Either a bundled example name (mild_postgres_ish, "
        "moderate_keyword_swap, aggressive_alien) or an importable "
        "'module.path:ATTR' Python reference to a DialectSpec.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List bundled spec examples and exit.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing dialect package on disk.",
    )
    p.add_argument(
        "--use-llm",
        "--force-llm",
        dest="use_llm",
        action="store_true",
        help="Run at least one LLM refinement pass on grammar AND lowering, "
        "even when the deterministic baseline already passes the battery. "
        "Regressions are rolled back to the deterministic baseline. Without "
        "this flag the deterministic emitter alone produces the package.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the LLM model used for refinement. Without "
        "--use-llm this is informational only.",
    )
    p.add_argument(
        "--lifecycle",
        default="generated",
        choices=[lc.value for lc in Lifecycle],
        help="Initial lifecycle state to record (default: generated).",
    )
    p.add_argument(
        "--require-battery-pass",
        action="store_true",
        help="Refuse to write the package if the parse / IR-equivalence "
        "battery still has failures after refinement.",
    )
    p.add_argument(
        "--dialects-dir",
        default=None,
        help="Override the output root (default: manysql/dialects/).",
    )
    return p


def _resolve_spec(ref: str) -> DialectSpec:
    """Accept either a bundled name or a 'module:attr' import path."""
    if ref in _BUNDLED_EXAMPLES:
        ref = _BUNDLED_EXAMPLES[ref]
    if ":" not in ref:
        raise SystemExit(
            f"error: '{ref}' is not a bundled example "
            f"({sorted(_BUNDLED_EXAMPLES)}) and is missing the "
            "'module:attribute' form for an arbitrary spec."
        )
    module_path, attr = ref.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(f"error: could not import {module_path!r}: {exc}") from exc
    try:
        spec = getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(
            f"error: module {module_path!r} has no attribute {attr!r}"
        ) from exc
    if not isinstance(spec, DialectSpec):
        raise SystemExit(
            f"error: {module_path}:{attr} is {type(spec).__name__}, "
            "expected a manysql.spec.dialect.DialectSpec"
        )
    return spec


def _maybe_llm_client(use_llm: bool, model: str | None):
    """Return an LLMClient if requested, or None to drive the deterministic path."""
    if not use_llm:
        return None
    try:
        from manysql.llm.client import LLMClient  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(f"error: cannot import manysql.llm.client: {exc}") from exc
    try:
        return LLMClient.from_env(default_model=model)
    except Exception as exc:
        raise SystemExit(
            f"error: --use-llm requested but no LLM backend configured: {exc}"
        ) from exc


def _print_examples(console: Console) -> None:
    table = Table(title="bundled DialectSpec examples")
    table.add_column("name")
    table.add_column("import path")
    for name, path in sorted(_BUNDLED_EXAMPLES.items()):
        table.add_row(name, path)
    console.print(table)


def _print_result(console: Console, result: PackageWriteResult) -> None:
    console.print(
        f"[green]Wrote[/green] dialect [bold]{result.name}[/bold] -> {result.path}"
    )
    console.print(f"  files: {', '.join(result.written_files)}")
    if result.grammar_result is not None:
        ok = "ok" if result.grammar_result.ok else "FAIL"
        attempts = len(result.grammar_result.attempts)
        sources = "+".join(sorted({a.source for a in result.grammar_result.attempts}))
        console.print(f"  grammar:  {ok}  ({attempts} attempt(s), {sources})")
    if result.lowering_result is not None:
        ok = "ok" if result.lowering_result.ok else "FAIL"
        attempts = len(result.lowering_result.attempts)
        sources = "+".join(sorted({a.source for a in result.lowering_result.attempts}))
        console.print(f"  lowering: {ok}  ({attempts} attempt(s), {sources})")
    console.print(
        "  next: [cyan]manysql-eval --backend synthetic --synthetic-dialect "
        f"{result.name} --dry-run[/cyan]"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()

    if args.list or args.spec is None:
        _print_examples(console)
        if args.spec is None and not args.list:
            console.print(
                "[yellow]error[/yellow]: pass a spec name or --list",
                style="dim",
            )
            return 2
        return 0

    spec = _resolve_spec(args.spec)
    llm_client = _maybe_llm_client(args.use_llm, args.model)
    root = Path(args.dialects_dir) if args.dialects_dir else DIALECTS_DIR

    try:
        result = write_dialect_package(
            spec,
            root,
            model=args.model,
            provider="llm" if llm_client else "deterministic",
            lifecycle=args.lifecycle,
            overwrite=args.overwrite,
            llm_client=llm_client,
            require_battery_pass=args.require_battery_pass,
            force_llm=bool(llm_client),
        )
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]hint:[/yellow] pass --overwrite to replace it.")
        return 1
    except BatteryError as exc:
        console.print(f"[red]battery failed:[/red] {exc}")
        return 1

    _print_result(console, result)

    # Sanity-check that the registry can load what we just wrote.
    try:
        DialectRegistry(root).load(result.name)
    except Exception as exc:
        console.print(f"[red]wrote files but registry failed to load them:[/red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
