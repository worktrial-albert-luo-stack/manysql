"""CLI: `python -m eval ...`.

Examples
--------

    # default: SQLite backend + OpenRouter
    python -m eval --model anthropic/claude-sonnet-4

    # OpenAI directly
    python -m eval --provider openai --model gpt-4o-mini

    # local vLLM serve endpoint
    python -m eval --provider vllm \\
        --vllm-base-url http://localhost:8000/v1 \\
        --model unsloth/Qwen3-4B-Instruct-2507

    # full Tinybird benchmark backend (requires TINYBIRD_* env vars)
    python -m eval --backend tinybird --provider openrouter \\
        --model anthropic/claude-sonnet-4

    # subset of questions, custom output, no LLM (sanity-check the runner)
    python -m eval --questions q01_count_stars,q02_top_starred_repos --dry-run

    # run only the first 5 questions (e.g. for smoke testing a new model)
    python -m eval --provider openai --model gpt-4o-mini --limit 5

    # parallelize LLM calls across 8 worker threads
    python -m eval --provider openai --model gpt-4o-mini -j 8

    # run against a manysql-generated dialect (auto-attaches a SQLite
    # reference executor for ground truth):
    uv run manysql-codegen gen mild_postgres_ish
    python -m eval --backend synthetic \\
        --synthetic-dialect mild_postgres_ish \\
        --provider openai --model gpt-4o-mini --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from eval.dataset.questions import select
from eval.executors import get_executor
from eval.llm import LLMClient
from eval.runner import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval",
        description=(
            "Run a tinybirdco-style LLM SQL benchmark against a pluggable "
            "execution backend (SQLite by default; Tinybird and synthetic "
            "manysql dialects also supported)."
        ),
    )

    # LLM provider
    p.add_argument(
        "--provider",
        choices=["openrouter", "openai", "vllm"],
        default="openrouter",
        help="LLM provider (default: openrouter)",
    )
    p.add_argument(
        "--model",
        required=False,
        help="Model id (e.g. 'anthropic/claude-sonnet-4', 'gpt-4o-mini', "
        "'unsloth/Qwen3-4B-Instruct-2507'). Required unless --dry-run.",
    )
    p.add_argument(
        "--vllm-base-url",
        default=None,
        help="OpenAI-compatible base URL for a local vLLM server, "
        "e.g. http://localhost:8000/v1",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Override the provider API key (else read from env).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max completion tokens (default: 2048).",
    )

    # Execution backend
    p.add_argument(
        "--backend",
        choices=["sqlite", "tinybird", "synthetic", "bird"],
        default="sqlite",
        help=(
            "SQL execution backend (default: sqlite). "
            "Choose 'bird' to evaluate against a subset of BIRD-SQL; "
            "see --bird-* flags for sizing the subset."
        ),
    )
    p.add_argument(
        "--sqlite-rows",
        type=int,
        default=5_000,
        help="Synthetic event rows for the SQLite backend (default: 5000).",
    )
    p.add_argument(
        "--sqlite-seed",
        type=int,
        default=0xDB,
        help="Seed for the SQLite synthetic dataset (default: 0xDB).",
    )
    p.add_argument(
        "--synthetic-dialect",
        default="_reference",
        help="Dialect id under manysql.dialects/ for the synthetic backend.",
    )
    p.add_argument(
        "--no-reference-executor",
        action="store_true",
        help="When --backend=synthetic, skip the auto-attached SQLite "
        "reference executor. The candidate dialect will then also be "
        "asked to run the (SQLite-flavored) reference SQL, which usually "
        "fails for anything beyond mild divergence. Useful only when "
        "you've authored dialect-specific reference SQL.",
    )

    # BIRD-specific options (used when --backend=bird).
    p.add_argument(
        "--bird-split",
        choices=["train", "dev"],
        default="dev",
        help="BIRD-SQL split to draw questions from (default: dev = "
        "1.5k community-reviewed questions; train = ~6.6k filtered).",
    )
    p.add_argument(
        "--bird-limit",
        type=int,
        default=50,
        help="How many BIRD questions to sample for this run "
        "(default: 50). Sampling is reproducible given --bird-seed.",
    )
    p.add_argument(
        "--bird-difficulties",
        default="simple,moderate",
        help="Comma-separated subset of {simple, moderate, challenging} "
        "to keep before sampling (default: simple,moderate).",
    )
    p.add_argument(
        "--bird-seed",
        type=int,
        default=0,
        help="PRNG seed for the BIRD subset (default: 0).",
    )
    p.add_argument(
        "--bird-db-dir",
        default=None,
        help="Override the directory holding <db_id>/<db_id>.sqlite. "
        "Defaults to $BIRD_DB_DIR or ~/.cache/manysql/bird/<split>/.",
    )
    p.add_argument(
        "--bird-sample-rows",
        type=int,
        default=3,
        help="Sample rows to include per table in the per-question "
        "prompt (default: 3; 0 disables).",
    )
    p.add_argument(
        "--bird-no-auto-download",
        action="store_true",
        help="Disable auto-download of the BIRD train zip (~5GB). "
        "Useful in air-gapped envs; pair with --bird-db-dir.",
    )

    # Run config
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max LLM retries on SQL errors (default: 2).",
    )
    p.add_argument(
        "--concurrency",
        "-j",
        type=int,
        default=1,
        help="Run this many questions in parallel via a thread pool "
        "(default: 1 = sequential). LLM calls are I/O-bound so threads "
        "give near-linear speedup until you hit your provider's rate "
        "limits or your local vLLM throughput.",
    )
    p.add_argument(
        "--questions",
        default=None,
        help="Comma-separated subset of question names. Default: all.",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Run at most this many questions (applied after --questions). "
        "Useful for quick smoke tests, e.g. `--limit 5`.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results. Default: results/<provider>_<model>.json",
    )
    p.add_argument(
        "--prompt-mode",
        choices=["plain", "tag"],
        default="plain",
        help=(
            "System-prompt format. 'plain' (default) asks for bare SQL "
            "with no markdown -- best for closed-source frontier models. "
            "'tag' asks the model to wrap output between <SQL>...</SQL> "
            "and matches what train/grpo_sql.py trains LoRAs on; use it "
            "when evaluating any LoRA produced by that pipeline. The "
            "extractor handles tags in either mode, so a tag-trained "
            "model run with --prompt-mode plain is still scored "
            "correctly -- the mode mostly affects whether the prompt "
            "*contradicts* what the model learned."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and per-question output.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM call; only run the reference SQL through the "
        "executor and report what *would* have been compared. Useful for "
        "verifying the dataset & schema without burning API credits.",
    )

    return p


def _default_output_path(provider: str, model: str, backend: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return Path("results") / f"{provider}_{safe_model}_{backend}.json"


def _build_bird_questions(
    args: argparse.Namespace, *, names: list[str] | None
) -> list[Any]:
    """Build the BIRD subset honoring `--bird-*` flags + `--questions/--limit`.

    Order of operations mirrors the github-events path:
      1. Sample N=`--bird-limit` questions from the chosen split,
         filtered by `--bird-difficulties` and seeded by `--bird-seed`.
      2. Optionally restrict to the names in `--questions`.
      3. Apply the global `--limit` cap (unusual for BIRD since
         `--bird-limit` already sizes the sample, but kept consistent
         with the other backends).
    """
    from eval.dataset.bird import select_bird  # noqa: PLC0415

    diffs = tuple(
        d.strip().lower()
        for d in args.bird_difficulties.split(",")
        if d.strip()
    )
    questions = select_bird(
        n_samples=args.bird_limit,
        split=args.bird_split,
        seed=args.bird_seed,
        difficulties=diffs,
        db_dir=args.bird_db_dir,
        sample_rows=args.bird_sample_rows,
        auto_download=not args.bird_no_auto_download,
    )
    if names is not None:
        by_name = {q.name: q for q in questions}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(
                f"--questions referenced unknown BIRD question name(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Names look like 'bird_<db_id>_<question_id>'; "
                "drop the filter or list with `manysql-eval --backend "
                "bird --dry-run` first."
            )
        questions = [by_name[n] for n in names]
    if args.limit is not None and args.limit >= 0:
        questions = questions[: args.limit]
    return questions


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # picks up .env from cwd; harmless if missing.

    args = _build_parser().parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    if not args.model:
        print("error: --model is required (or use --dry-run)", file=sys.stderr)
        return 2

    base_url = None
    if args.provider == "vllm":
        base_url = args.vllm_base_url or os.getenv("VLLM_BASE_URL")
        if not base_url:
            print(
                "error: provider=vllm requires --vllm-base-url "
                "(or $VLLM_BASE_URL)",
                file=sys.stderr,
            )
            return 2

    executor_kwargs: dict[str, Any] = {}
    if args.backend == "sqlite":
        executor_kwargs = {
            "n_rows": args.sqlite_rows,
            "seed": args.sqlite_seed,
        }
    elif args.backend == "synthetic":
        executor_kwargs = {"dialect": args.synthetic_dialect}

    executor = get_executor(args.backend, **executor_kwargs)

    reference_executor = None
    if args.backend == "synthetic" and not args.no_reference_executor:
        reference_executor = get_executor(
            "sqlite",
            n_rows=args.sqlite_rows,
            seed=args.sqlite_seed,
        )

    names = (
        [q.strip() for q in args.questions.split(",") if q.strip()]
        if args.questions
        else None
    )
    if args.backend == "bird":
        questions = _build_bird_questions(args, names=names)
    else:
        questions = (
            select(names, limit=args.limit)
            if names is not None or args.limit is not None
            else None
        )

    output = (
        Path(args.output)
        if args.output
        else _default_output_path(args.provider, args.model, args.backend)
    )

    with LLMClient(
        provider=args.provider,
        model=args.model,
        base_url=base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ) as llm:
        run_benchmark(
            llm=llm,
            executor=executor,
            reference_executor=reference_executor,
            questions=questions,
            max_retries=args.max_retries,
            concurrency=args.concurrency,
            output_path=output,
            quiet=args.quiet,
            prompt_mode=args.prompt_mode,
        )
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    """Run reference SQL through the executor and print row counts.

    Doesn't call any LLM. Use it to (a) verify your backend connects, and
    (b) sanity-check that the seed dataset produces non-empty answers.
    """
    console = Console()
    executor_kwargs: dict[str, Any] = {}
    if args.backend == "sqlite":
        executor_kwargs = {
            "n_rows": args.sqlite_rows,
            "seed": args.sqlite_seed,
        }
    elif args.backend == "synthetic":
        executor_kwargs = {"dialect": args.synthetic_dialect}
    executor = get_executor(args.backend, **executor_kwargs)
    executor.setup()
    try:
        names = (
            [q.strip() for q in args.questions.split(",") if q.strip()]
            if args.questions
            else None
        )
        if args.backend == "bird":
            questions = _build_bird_questions(args, names=names)
        else:
            questions = select(names, limit=args.limit)

        table = Table(title=f"dry-run on backend={executor.name}")
        table.add_column("question")
        table.add_column("rows", justify="right")
        table.add_column("status")

        for q in questions:
            dialect = executor.dialect_label().lower()
            ref_sql = next(
                (sql for k, sql in q.reference_sql.items() if k.lower() in dialect),
                None,
            )
            ref_source = "match"
            if ref_sql is None and "sqlite" in q.reference_sql:
                # Synthetic dialects don't have their own reference SQL yet;
                # fall back to SQLite reference text the same way the runner
                # does. Useful for confirming the dialect engine can parse
                # the SQLite surface of mild dialects.
                ref_sql = q.reference_sql["sqlite"]
                ref_source = "sqlite-fallback"
            if ref_sql is None:
                table.add_row(q.name, "-", "[yellow]no reference SQL for dialect[/yellow]")
                continue
            res = executor.execute(ref_sql, question=q)
            if res.success:
                tag = (
                    "[green]ok[/green]"
                    if ref_source == "match"
                    else "[green]ok (sqlite-fallback)[/green]"
                )
                table.add_row(q.name, str(len(res.rows)), tag)
            else:
                table.add_row(q.name, "-", f"[red]{res.error}[/red]")
        console.print(table)
    finally:
        executor.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
