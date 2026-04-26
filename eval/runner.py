"""Eval runner: glue between LLM, executor, validator, and the question suite.

Mirrors the control flow of `tinybirdco/llm-benchmark/src/benchmark/index.ts`:

  1. For each question, ask the LLM to write SQL.
  2. Execute the SQL on the chosen backend.
  3. If it errors, retry up to N times with the error fed back as context.
  4. Run the reference SQL through the same backend (once per question)
     and compare the two result sets via `eval.validator`.
  5. Aggregate per-model stats and dump to JSON.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from eval.dataset.questions import Question, select
from eval.executors.base import ExecResult, SqlExecutor
from eval.llm import LLMClient
from eval.prompt import build_system_prompt, extract_sql
from eval.validator import ComparisonResult, compare_results

DEFAULT_MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 1

console = Console(stderr=True)


@dataclass
class Attempt:
    sql: str
    llm: dict[str, Any]
    exec_result: dict[str, Any]


@dataclass
class QuestionResult:
    name: str
    prompt: str
    final_sql: str | None
    final_exec: dict[str, Any] | None
    attempts: list[Attempt] = field(default_factory=list)
    reference_sql: str | None = None
    reference_exec: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class BenchmarkSummary:
    total: int
    matched: int
    exact_matched: int
    numeric_matched: int
    avg_first_attempt_success: float
    avg_attempts: float
    avg_total_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRun:
    provider: str
    model: str
    backend: str
    dialect: str
    questions: list[QuestionResult]
    summary: BenchmarkSummary
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "backend": self.backend,
            "dialect": self.dialect,
            "summary": self.summary.to_dict(),
            "questions": [q.to_dict() for q in self.questions],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def run_benchmark(
    *,
    llm: LLMClient,
    executor: SqlExecutor,
    questions: list[Question] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    reference_executor: SqlExecutor | None = None,
    output_path: str | Path | None = None,
    quiet: bool = False,
) -> BenchmarkRun:
    """Run the suite. Returns a `BenchmarkRun` and optionally writes JSON.

    Args:
        concurrency: number of questions to evaluate in parallel. The work
            per question is dominated by a network LLM call, so a thread
            pool is sufficient (and SQLite + httpx are both safe for
            concurrent use within a single process). ``1`` (the default)
            preserves the historical sequential behavior.
        reference_executor: optional separate backend used to compute
            ground truth from the question suite's reference SQL. When
            ``None`` (the default), the candidate ``executor`` is also
            used as the reference -- this is what you want for SQLite/
            Tinybird where the reference SQL is in the same dialect. For
            generated synthetic dialects, pass an SQLite executor here so
            ground truth comes from the SQLite reference SQL while the
            LLM's SQL is judged through the dialect engine.
    """
    qs = list(questions) if questions is not None else select()
    if not qs:
        raise ValueError("no questions to run")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    concurrency = min(concurrency, len(qs))

    ref_exec_backend = reference_executor or executor

    started = _utc_now()
    if not quiet:
        ref_blurb = (
            ""
            if reference_executor is None
            else f"  reference={ref_exec_backend.name}/{ref_exec_backend.dialect_label()}"
        )
        console.print(
            f"[bold]Eval[/bold]: {len(qs)} questions  "
            f"provider={llm.provider} model={llm.model}  "
            f"backend={executor.name} dialect={executor.dialect_label()}  "
            f"concurrency={concurrency}{ref_blurb}"
        )

    executor.setup()
    if reference_executor is not None and reference_executor is not executor:
        reference_executor.setup()
    system_prompt = build_system_prompt(executor)

    progress_cm: Any
    if quiet:
        progress_cm = _NullProgress()
    else:
        progress_cm = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    try:
        with progress_cm as progress:
            task = progress.add_task("running", total=len(qs)) if not quiet else None
            if concurrency == 1:
                results = _run_sequential(
                    qs,
                    llm=llm,
                    executor=executor,
                    reference_executor=ref_exec_backend,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    progress=progress,
                    task=task,
                    quiet=quiet,
                )
            else:
                results = _run_parallel(
                    qs,
                    llm=llm,
                    executor=executor,
                    reference_executor=ref_exec_backend,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    concurrency=concurrency,
                    progress=progress,
                    task=task,
                    quiet=quiet,
                )
    finally:
        executor.teardown()
        if reference_executor is not None and reference_executor is not executor:
            reference_executor.teardown()

    summary = _summarize(results)
    finished = _utc_now()
    run = BenchmarkRun(
        provider=llm.provider,
        model=llm.model,
        backend=executor.name,
        dialect=executor.dialect_label(),
        questions=results,
        summary=summary,
        started_at=started,
        finished_at=finished,
    )

    if not quiet:
        _print_summary(run)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run.to_dict(), indent=2, default=str))
        if not quiet:
            console.print(f"[green]Results written to {out}[/green]")

    return run


# ----- internals -----------------------------------------------------------


def _run_sequential(
    qs: list[Question],
    *,
    llm: LLMClient,
    executor: SqlExecutor,
    reference_executor: SqlExecutor,
    system_prompt: str,
    max_retries: int,
    progress: Any,
    task: Any,
    quiet: bool,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for q in qs:
        qr = _run_one_question(
            q,
            llm=llm,
            executor=executor,
            reference_executor=reference_executor,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )
        results.append(qr)
        if not quiet:
            _print_question_summary(qr)
            progress.advance(task)
    return results


def _run_parallel(
    qs: list[Question],
    *,
    llm: LLMClient,
    executor: SqlExecutor,
    reference_executor: SqlExecutor,
    system_prompt: str,
    max_retries: int,
    concurrency: int,
    progress: Any,
    task: Any,
    quiet: bool,
) -> list[QuestionResult]:
    """Fan out questions over a thread pool, then re-sort into input order.

    LLM calls dominate per-question time and are I/O-bound, so threads are
    sufficient. ``httpx.Client`` is documented as thread-safe for
    concurrent ``request()`` calls, and SQLite (with
    ``check_same_thread=False``) serializes statements at the connection
    level — fine since the SQL eval is microseconds vs the LLM seconds.
    """
    print_lock = threading.Lock()
    results_by_idx: dict[int, QuestionResult] = {}

    def _work(idx_q: tuple[int, Question]) -> tuple[int, QuestionResult]:
        idx, q = idx_q
        qr = _run_one_question(
            q,
            llm=llm,
            executor=executor,
            reference_executor=reference_executor,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )
        return idx, qr

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_work, (i, q)) for i, q in enumerate(qs)]
        for fut in as_completed(futures):
            idx, qr = fut.result()
            results_by_idx[idx] = qr
            if not quiet:
                with print_lock:
                    _print_question_summary(qr)
                    progress.advance(task)

    return [results_by_idx[i] for i in range(len(qs))]


def _run_one_question(
    q: Question,
    *,
    llm: LLMClient,
    executor: SqlExecutor,
    reference_executor: SqlExecutor,
    system_prompt: str,
    max_retries: int,
) -> QuestionResult:
    user = q.prompt
    attempts: list[Attempt] = []

    for retry in range(max_retries + 1):
        llm_resp = llm.chat(system=system_prompt, user=user)
        if llm_resp.error:
            attempts.append(
                Attempt(
                    sql="",
                    llm=llm_resp.to_dict(),
                    exec_result=ExecResult(
                        success=False,
                        error="LLM call failed",
                        backend=executor.name,
                    ).to_dict(),
                )
            )
            break  # don't keep retrying on transport errors

        sql = extract_sql(llm_resp.text)
        exec_res = executor.execute(sql)
        attempts.append(
            Attempt(
                sql=sql,
                llm=llm_resp.to_dict(),
                exec_result=exec_res.to_dict(),
            )
        )

        if exec_res.success:
            break

        # Retry: feed the error back as part of a fresh user turn.
        if retry == max_retries:
            break
        user = (
            f"I asked: {q.prompt}\n\n"
            f"You produced this SQL:\n{sql}\n\n"
            f"It failed with this error:\n{exec_res.error}\n\n"
            "Please rewrite the SQL so it executes correctly."
        )

    final_attempt = attempts[-1] if attempts else None
    final_sql = final_attempt.sql if final_attempt else None
    final_exec = final_attempt.exec_result if final_attempt else None
    final_llm_error = (
        final_attempt.llm.get("error") if final_attempt else None
    )

    ref_sql, ref_exec, comp = _evaluate_reference(q, reference_executor, final_exec)

    # If the LLM call itself failed (bad request, network, etc.), overwrite
    # the comparison detail so downstream readers see the actual root cause
    # instead of a misleading "results do not match".
    if comp is not None and final_llm_error:
        comp.detail = f"LLM error: {final_llm_error}"

    return QuestionResult(
        name=q.name,
        prompt=q.prompt,
        final_sql=final_sql,
        final_exec=final_exec,
        attempts=attempts,
        reference_sql=ref_sql,
        reference_exec=ref_exec.to_dict() if ref_exec else None,
        comparison=comp.to_dict() if comp else None,
        error=final_llm_error,
    )


def _evaluate_reference(
    q: Question,
    executor: SqlExecutor,
    final_exec: dict[str, Any] | None,
) -> tuple[str | None, ExecResult | None, ComparisonResult | None]:
    dialect = executor.dialect_label().lower()
    ref_sql: str | None = None
    for key, sql in q.reference_sql.items():
        if key.lower() in dialect:
            ref_sql = sql
            break
    # Synthetic dialects don't (yet) have their own reference SQL keyed in
    # the question suite; fall back to SQLite reference text since the
    # caller is expected to pass a SQLite-backed `reference_executor` in
    # that case.
    if ref_sql is None and "sqlite" in q.reference_sql:
        ref_sql = q.reference_sql["sqlite"]
    if ref_sql is None:
        return None, None, None

    ref_exec = executor.execute(ref_sql)
    if not ref_exec.success:
        # Bench infra bug, not the LLM's fault. Surface it but don't crash.
        return ref_sql, ref_exec, None

    if not final_exec or not final_exec.get("success"):
        comp = compare_results(ref_exec.rows, [])
    else:
        comp = compare_results(ref_exec.rows, final_exec.get("rows", []))
    return ref_sql, ref_exec, comp


def _summarize(results: list[QuestionResult]) -> BenchmarkSummary:
    n = len(results)
    matched = sum(1 for r in results if (r.comparison or {}).get("matches"))
    exact = sum(1 for r in results if (r.comparison or {}).get("exact_match"))
    numeric = sum(1 for r in results if (r.comparison or {}).get("numeric_match"))

    first_ok = sum(
        1
        for r in results
        if r.attempts and r.attempts[0].exec_result.get("success")
    )
    attempt_counts = [len(r.attempts) for r in results]
    durations = [
        sum(a.llm.get("duration_s", 0.0) for a in r.attempts) for r in results
    ]
    return BenchmarkSummary(
        total=n,
        matched=matched,
        exact_matched=exact,
        numeric_matched=numeric,
        avg_first_attempt_success=first_ok / n if n else 0.0,
        avg_attempts=sum(attempt_counts) / n if n else 0.0,
        avg_total_duration_s=sum(durations) / n if n else 0.0,
    )


def _print_question_summary(qr: QuestionResult) -> None:
    final_ok = bool((qr.final_exec or {}).get("success"))
    matched = bool((qr.comparison or {}).get("matches"))
    if qr.error:
        status = "[red]llm error[/red]"
    elif final_ok and matched:
        status = "[green]ok[/green]"
    elif final_ok:
        status = "[yellow]ran but mismatched[/yellow]"
    else:
        status = "[red]sql error[/red]"
    detail = (qr.comparison or {}).get("detail", "")
    if qr.error and not detail.startswith("LLM error"):
        detail = f"LLM error: {qr.error}"
    console.print(f"  {status}  {qr.name}  ({len(qr.attempts)} attempt(s))  {detail}")


def _print_summary(run: BenchmarkRun) -> None:
    s = run.summary
    console.print()
    console.print(f"[bold]Summary[/bold]  {run.provider}/{run.model}  ({run.backend})")
    console.print(
        f"  matched: {s.matched}/{s.total}  "
        f"(exact={s.exact_matched}, numeric={s.numeric_matched})"
    )
    console.print(
        f"  first-attempt success: {s.avg_first_attempt_success:.0%}   "
        f"avg attempts: {s.avg_attempts:.2f}   "
        f"avg total LLM time: {s.avg_total_duration_s:.2f}s"
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _NullProgress:
    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def add_task(self, *_: Any, **__: Any) -> None:
        return None

    def advance(self, *_: Any, **__: Any) -> None:
        return None


__all__ = [
    "Attempt",
    "BenchmarkRun",
    "BenchmarkSummary",
    "QuestionResult",
    "run_benchmark",
]
