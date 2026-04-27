"""CLI: ``python -m manysql.codegen`` / ``manysql-codegen``.

Two subcommands:

- ``gen`` materializes ONE dialect package from a ``DialectSpec`` (bundled
  example or ``module.path:ATTR``). This is the original single-spec
  workflow, just under a subcommand name.
- ``batch`` runs an outer-loop campaign: design N diverse specs from a
  free-form prior + structured knobs and fan them through the inner
  pipeline. Drives ``manysql.codegen.batch.run_campaign``.

Examples
--------

    # list bundled spec examples
    manysql-codegen --list

    # generate a single dialect via the deterministic emitter
    manysql-codegen gen mild_postgres_ish

    # ALWAYS run at least one LLM refinement pass on top of the deterministic
    # baseline. Uses OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY.
    manysql-codegen gen aggressive_alien --use-llm

    # arbitrary spec from an importable Python attribute
    manysql-codegen gen mypkg.specs:MY_SPEC --overwrite

    # design 5 diverse dialects in one campaign
    manysql-codegen batch --n 5 --prior "variants between mssql and snowflake"
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from manysql.codegen.batch import (
    CAMPAIGNS_DIRNAME,
    CampaignBrief,
    CampaignConfig,
    CampaignReporter,
    CampaignResult,
    LedgerEntry,
    THEME_CHOICES,
    run_campaign,
)
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
    "snowflake_clone": "manysql.spec.examples.snowflake_clone:SNOWFLAKE_CLONE",
    "sqlite_clone": "manysql.spec.examples.sqlite_clone:SQLITE_CLONE",
    "postgres_clone": "manysql.spec.examples.postgres_clone:POSTGRES_CLONE",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manysql-codegen",
        description=(
            "Generate manysql dialect packages. Use `gen` for a single "
            "spec, `batch` for a multi-dialect campaign. Run `--list` to "
            "see bundled example specs."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List bundled spec examples and exit. Works without a subcommand.",
    )
    sub = p.add_subparsers(dest="cmd")

    _add_gen_parser(sub)
    _add_batch_parser(sub)

    return p


def _add_gen_parser(sub: argparse._SubParsersAction) -> None:
    gen = sub.add_parser(
        "gen",
        help="Materialize a single dialect package from a DialectSpec.",
        description=(
            "Run the codegen pipeline on one DialectSpec (bundled or "
            "imported). The deterministic emitter is the default; "
            "`--use-llm` adds an LLM refinement pass that is rolled back "
            "if it regresses the parse/IR battery."
        ),
    )
    gen.add_argument(
        "spec",
        help="Either a bundled example name (mild_postgres_ish, "
        "moderate_keyword_swap, aggressive_alien) or an importable "
        "'module.path:ATTR' Python reference to a DialectSpec.",
    )
    gen.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing dialect package on disk.",
    )
    gen.add_argument(
        "--use-llm",
        "--force-llm",
        dest="use_llm",
        action="store_true",
        help="Run at least one LLM refinement pass on grammar AND lowering, "
        "even when the deterministic baseline already passes the battery. "
        "Regressions are rolled back to the deterministic baseline.",
    )
    gen.add_argument(
        "--model",
        default=None,
        help="Override the LLM model used for refinement. Without "
        "--use-llm this is informational only.",
    )
    gen.add_argument(
        "--lifecycle",
        default="generated",
        choices=[lc.value for lc in Lifecycle],
        help="Initial lifecycle state to record (default: generated).",
    )
    gen.add_argument(
        "--require-battery-pass",
        action="store_true",
        help="Refuse to write the package if the parse / IR-equivalence "
        "battery still has failures after refinement.",
    )
    gen.add_argument(
        "--grammar-max-iterations",
        type=int,
        default=3,
        help="Max LLM refinement rounds for the grammar agent "
        "(default: 3). Each round is one extra LLM call per spec.",
    )
    gen.add_argument(
        "--lowering-max-iterations",
        type=int,
        default=3,
        help="Max LLM refinement rounds for the lowering agent "
        "(default: 3). Bumping this gives the agent more shots at "
        "fixing IR-equivalence failures, at proportional API cost.",
    )
    gen.add_argument(
        "--dialects-dir",
        default=None,
        help="Override the output root (default: manysql/dialects/).",
    )


def _add_batch_parser(sub: argparse._SubParsersAction) -> None:
    batch = sub.add_parser(
        "batch",
        help="Design N diverse dialects in one campaign.",
        description=(
            "Outer-loop orchestrator: design N specs sequentially from a "
            "free-form prior and structured knobs, then fan them through "
            "the codegen pipeline in parallel. Writes a campaign manifest "
            "under <dialects-dir>/_campaigns/<id>.json. Requires an LLM "
            "backend (OPENAI/OPENROUTER/ANTHROPIC API key)."
        ),
    )
    batch.add_argument(
        "--n", type=int, required=True, help="Number of dialects to design."
    )
    batch.add_argument(
        "--prior",
        default=None,
        help="Free-form description of the campaign vibe, e.g. "
        "'variants between mssql and snowflake'. Expanded into a "
        "structured campaign brief in one upfront LLM call.",
    )
    batch.add_argument(
        "--theme",
        choices=list(THEME_CHOICES),
        default="mixed",
        help="Divergence target. 'mixed' rotates mild/moderate/aggressive "
        "across slots; explicit values use the same level for every slot.",
    )
    batch.add_argument(
        "--inspired-by",
        default="",
        help="Comma-separated real-world dialects to draw from "
        "(e.g. 'mysql,kdb,snowflake').",
    )
    batch.add_argument(
        "--exclude-knobs",
        default="",
        help="Comma-separated DialectSpec field names that workers must "
        "not change. Always wins over the prior.",
    )
    batch.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Recorded in the manifest for reproducibility tracking.",
    )
    batch.add_argument(
        "--model",
        default=None,
        help="Override the LLM model used for both brief expansion and "
        "per-dialect design.",
    )
    batch.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Worker count for the inner-pipeline fan-out (default: 4).",
    )
    batch.add_argument(
        "--require-battery-pass",
        action="store_true",
        help="Reject any dialect whose parse / IR battery still fails "
        "after refinement.",
    )
    batch.add_argument(
        "--grammar-max-iterations",
        type=int,
        default=3,
        help="Max LLM refinement rounds for the grammar agent on each "
        "spec (default: 3).",
    )
    batch.add_argument(
        "--lowering-max-iterations",
        type=int,
        default=3,
        help="Max LLM refinement rounds for the lowering agent on each "
        "spec (default: 3). Higher caps cost proportionally more.",
    )
    batch.add_argument(
        "--dialects-dir",
        default=None,
        help="Override the output root (default: manysql/dialects/).",
    )


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
    return _build_llm_client(model)


def _build_llm_client(model: str | None):
    """Return an LLMClient or raise SystemExit with a friendly message."""
    try:
        from manysql.llm.client import LLMClient  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(f"error: cannot import manysql.llm.client: {exc}") from exc
    try:
        return LLMClient.from_env(default_model=model)
    except Exception as exc:
        raise SystemExit(f"error: no LLM backend configured: {exc}") from exc


def _print_examples(console: Console) -> None:
    table = Table(title="bundled DialectSpec examples")
    table.add_column("name")
    table.add_column("import path")
    for name, path in sorted(_BUNDLED_EXAMPLES.items()):
        table.add_row(name, path)
    console.print(table)


def _print_gen_result(console: Console, result: PackageWriteResult) -> None:
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


class RichCampaignReporter(CampaignReporter):
    """Live progress for ``manysql-codegen batch`` runs.

    Emits one line per stage transition so a long-running campaign is
    legible in a plain terminal. The package phase runs concurrently, so
    package events may interleave - this is fine, the messages are
    self-identifying.
    """

    def __init__(self, console: Console) -> None:
        self._console = console

    def on_campaign_start(
        self, *, config: CampaignConfig, model: str
    ) -> None:
        self._console.print(
            f"[bold cyan]campaign[/bold cyan] starting: "
            f"n={config.n}, theme={config.theme}, "
            f"prior={(config.prior or '')!r}, model={model}"
        )

    def on_brief_start(self) -> None:
        self._console.print("[dim]> expanding campaign brief...[/dim]")

    def on_brief_done(
        self, *, brief: CampaignBrief, elapsed_s: float
    ) -> None:
        inspirations = ", ".join(brief.inspirations) or "-"
        axes = ", ".join(brief.suggested_axes[:5]) or "-"
        if len(brief.suggested_axes) > 5:
            axes += ", ..."
        self._console.print(
            f"[green]ok[/green] brief expanded in {elapsed_s:.1f}s "
            f"[dim](inspirations: {inspirations}; axes: {axes})[/dim]"
        )

    def on_design_phase_start(self, *, schedule: list[str]) -> None:
        self._console.print(
            f"[bold cyan]design[/bold cyan] {len(schedule)} specs "
            f"[dim](schedule: {', '.join(schedule)})[/dim]"
        )

    def on_design_slot_attempt(
        self,
        *,
        slot: int,
        total: int,
        target_divergence: str,
        attempt: int,
    ) -> None:
        if attempt == 1:
            self._console.print(
                f"  [dim]> slot {slot + 1}/{total} "
                f"[{target_divergence}] designing...[/dim]"
            )
        else:
            self._console.print(
                f"  [yellow]retry[/yellow] slot {slot + 1}/{total} "
                f"[{target_divergence}] (attempt {attempt})"
            )

    def on_design_slot_done(
        self,
        *,
        slot: int,
        total: int,
        entry: LedgerEntry,
        divergence: str,
        elapsed_s: float,
    ) -> None:
        axes = ", ".join(entry.primary_axes[:3]) or "-"
        if len(entry.primary_axes) > 3:
            axes += ", ..."
        self._console.print(
            f"  [green]ok[/green] slot {slot + 1}/{total} [{divergence}] "
            f"-> [bold]{entry.name}[/bold] "
            f"[dim](axes: {axes}; {elapsed_s:.1f}s)[/dim]"
        )

    def on_design_slot_failed(
        self,
        *,
        slot: int,
        total: int,
        target_divergence: str,
        reason: str,
    ) -> None:
        self._console.print(
            f"  [red]fail[/red] slot {slot + 1}/{total} [{target_divergence}]: "
            f"[dim]{_truncate(reason, 120)}[/dim]"
        )

    def on_package_phase_start(
        self, *, n: int, max_concurrency: int
    ) -> None:
        self._console.print(
            f"[bold cyan]package[/bold cyan] {n} dialects "
            f"[dim](max_concurrency={max_concurrency})[/dim]"
        )

    def on_package_done(
        self,
        *,
        name: str,
        summary: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        bits = []
        for stage in ("grammar_ok", "lowering_ok"):
            value = summary.get(stage)
            if value is True:
                bits.append(f"{stage[:-3]}=ok")
            elif value is False:
                bits.append(f"[yellow]{stage[:-3]}=warn[/yellow]")
        detail = ", ".join(bits) if bits else "-"
        self._console.print(
            f"  [green]ok[/green] packaged [bold]{name}[/bold] "
            f"[dim]({detail}; {elapsed_s:.1f}s)[/dim]"
        )

    def on_package_failed(
        self,
        *,
        name: str,
        reason: str,
        elapsed_s: float,
    ) -> None:
        self._console.print(
            f"  [red]fail[/red] packaging [bold]{name}[/bold] "
            f"[dim]({elapsed_s:.1f}s)[/dim]: [dim]{_truncate(reason, 120)}[/dim]"
        )

    def on_manifest_written(self, *, path: Path) -> None:
        self._console.print(f"[dim]> manifest -> {path}[/dim]")

    def on_campaign_done(self, *, result: CampaignResult) -> None:
        self._console.print(
            f"[bold cyan]done[/bold cyan] drafted={len(result.drafted)}, "
            f"packaged={len(result.packaged)}, "
            f"spec_failed={len(result.failed_specs)}, "
            f"package_failed={len(result.failed_packages)}"
        )

    def on_interrupted(self, *, stage: str) -> None:
        self._console.print(
            f"[yellow]interrupted[/yellow] during {stage}; finalizing "
            "partial manifest. Already-running LLM HTTP calls will keep "
            "running until they finish or hit their timeout."
        )


def _truncate(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "..."


def _print_campaign_result(
    console: Console, result: CampaignResult, dialects_root: Path
) -> None:
    table = Table(title=f"campaign {result.id}")
    table.add_column("slot", justify="right")
    table.add_column("name")
    table.add_column("divergence")
    table.add_column("axes")
    table.add_column("status")

    packaged_names = {p.name for p in result.packaged}
    failed_pkg_names = {f["name"]: f for f in result.failed_packages}

    for slot, (spec, entry) in enumerate(result.drafted):
        if entry.name in packaged_names:
            status = "[green]packaged[/green]"
        elif entry.name in failed_pkg_names:
            reason = failed_pkg_names[entry.name].get("reason", "")
            status = f"[red]package_failed[/red] ({reason[:40]})"
        else:
            status = "[yellow]drafted[/yellow]"
        table.add_row(
            str(slot),
            entry.name,
            spec.divergence.value,
            ", ".join(entry.primary_axes) or "-",
            status,
        )
    for f in result.failed_specs:
        table.add_row(
            str(f.get("slot", "?")),
            "-",
            str(f.get("target_divergence", "?")),
            "-",
            f"[red]spec_failed[/red] ({str(f.get('reason', ''))[:40]})",
        )
    console.print(table)
    console.print(
        f"manifest: [cyan]{dialects_root / CAMPAIGNS_DIRNAME / (result.id + '.json')}[/cyan]"
    )
    console.print(
        f"summary: drafted={len(result.drafted)}, packaged={len(result.packaged)}, "
        f"spec_failed={len(result.failed_specs)}, package_failed={len(result.failed_packages)}"
    )


def _run_gen(args: argparse.Namespace, console: Console) -> int:
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
            grammar_max_iterations=args.grammar_max_iterations,
            lowering_max_iterations=args.lowering_max_iterations,
        )
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]hint:[/yellow] pass --overwrite to replace it.")
        return 1
    except BatteryError as exc:
        console.print(f"[red]battery failed:[/red] {exc}")
        return 1

    _print_gen_result(console, result)

    try:
        DialectRegistry(root).load(result.name)
    except Exception as exc:  # noqa: BLE001 - surface load errors clearly
        console.print(f"[red]wrote files but registry failed to load them:[/red] {exc}")
        return 1
    return 0


def _run_batch(args: argparse.Namespace, console: Console) -> int:
    if args.n <= 0:
        console.print("[red]error[/red]: --n must be positive")
        return 2

    inspired_by = tuple(_split_csv(args.inspired_by))
    exclude_knobs = tuple(_split_csv(args.exclude_knobs))

    config = CampaignConfig(
        n=args.n,
        prior=args.prior,
        theme=args.theme,
        inspired_by=inspired_by,
        exclude_knobs=exclude_knobs,
        seed=args.seed,
        model=args.model,
        max_concurrency=args.max_concurrency,
        require_battery_pass=args.require_battery_pass,
        grammar_max_iterations=args.grammar_max_iterations,
        lowering_max_iterations=args.lowering_max_iterations,
    )
    root = Path(args.dialects_dir) if args.dialects_dir else DIALECTS_DIR

    llm = _build_llm_client(args.model)
    reporter = RichCampaignReporter(console)
    interrupted = False
    try:
        try:
            result = run_campaign(
                config, llm=llm, dialects_root=root, reporter=reporter
            )
        except KeyboardInterrupt:
            interrupted = True
            # run_campaign re-raises after writing a partial manifest;
            # the manifest path was already announced via the reporter.
            console.print(
                "\n[yellow]campaign interrupted by user[/yellow]; "
                "see the partial manifest above."
            )
            return 130
    finally:
        llm.close()

    console.print()
    _print_campaign_result(console, result, root)
    if interrupted:
        return 130
    if not result.drafted and not result.packaged:
        return 1
    return 0


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.list:
        _print_examples(console)
        return 0

    if args.cmd == "gen":
        return _run_gen(args, console)
    if args.cmd == "batch":
        return _run_batch(args, console)

    parser.print_help()
    console.print(
        "\n[yellow]error[/yellow]: pick a subcommand "
        "([cyan]gen[/cyan] or [cyan]batch[/cyan]) or pass [cyan]--list[/cyan]."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
