"""Serve a LoRA adapter atop a base model in vLLM, then run one or more
eval configs against the served endpoint.

This is the connective tissue between training and evaluation: after
``train/grpo_sql.py`` writes a LoRA adapter, you usually want to
benchmark that checkpoint across several dialects/datasets. Spinning
up vLLM costs ~30-60s on H100 (model load + CUDA graph capture) so
doing one server start per ``python -m eval`` invocation is wasteful.
This wrapper pays the cost once and reuses the server across runs.

Workflow
--------

After training writes ``outputs/grpo_qwen3_4b_sql/lora/``::

    python -m eval.serve_lora \\
        --base-model unsloth/Qwen3-4B-Instruct-2507 \\
        --lora-path outputs/grpo_qwen3_4b_sql/lora \\
        --dialects aggressive_alien,mild_postgres_ish,tsql_ish \\
        --backend synthetic --limit 20 -j 4

That spawns ``vllm serve <base> --enable-lora --lora-modules
<name>=<path>``, waits for ``/health`` to return 200, then runs three
``python -m eval`` invocations against ``http://localhost:8000/v1``
with model id ``<name>`` (one per dialect) and tears the server down
when finished or interrupted.

For non-synthetic backends (or running the same dialect with multiple
question subsets), point ``--runs`` at a JSON file::

    [
      {"backend": "sqlite", "limit": 50, "concurrency": 4},
      {"backend": "synthetic", "synthetic_dialect": "aggressive_alien",
       "questions": "q01_count_stars,q05_top_repos_by_year_since_2015"}
    ]

Each entry is a dict of ``--<key>`` overrides forwarded to
``python -m eval``. Underscores are translated to dashes
(``synthetic_dialect`` -> ``--synthetic-dialect``) and ``True``/``False``
values become bare flags / their absence.

Trailing args after ``--`` are forwarded verbatim to **every** eval
invocation so any flag the eval CLI accepts still works::

    python -m eval.serve_lora --lora-path ... --dialects a,b -- \\
        --temperature 0.2 --max-tokens 4096

Prompt format
-------------

This wrapper defaults each eval run to ``--prompt-mode tag`` (the
``<SQL>...</SQL>`` protocol that ``train/grpo_sql.py`` trains on).
Override with ``--prompt-mode plain`` if you're benching a model
that was *not* trained on the tag protocol (or set ``prompt_mode``
per-entry in a ``--runs`` JSON file).

Server lifecycle
----------------

The vLLM subprocess inherits stdio so you see boot logs in real time.
On Ctrl-C / completion / eval failure we send SIGTERM, wait briefly,
then SIGKILL if it hasn't exited. Pass ``--keep-server`` to leave the
server running after all evals finish (useful for ad-hoc curl probes
or rerunning the eval CLI manually); the script then prints the
endpoint and the lora-name and exits 0 without killing vllm.

Base-model baseline (no LoRA)
-----------------------------

Omit ``--lora-path`` to serve the bare base model. The vllm command
then drops ``--enable-lora``/``--lora-modules`` and the eval CLI is
called with ``--model <base-model>``. Useful for establishing a
pre-training baseline against which you can compare your LoRA
checkpoints on the same dialects::

    python -m eval.serve_lora \\
        --base-model unsloth/Qwen3-4B-Instruct-2507 \\
        --dialects aggressive_alien,mild_postgres_ish --limit 20

Pre-existing server
-------------------

If you've already started vllm yourself (e.g. via ``vllm serve`` in a
tmux pane), pass ``--no-server`` and the script will skip launch +
teardown and just dispatch eval runs against ``--vllm-base-url``
(default ``http://localhost:8000/v1``). With ``--no-server`` and
``--lora-path`` set, ``--lora-name`` must match the module name the
existing server exposes; without ``--lora-path`` the eval is
dispatched against ``--base-model`` (so that must match too).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_HEALTH_TIMEOUT_S = 600.0
DEFAULT_HEALTH_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """One ``python -m eval`` invocation against the served endpoint.

    ``overrides`` is a dict of ``--<flag>`` -> value pairs forwarded
    verbatim to the eval CLI. Underscored keys are translated to
    dashed flags. Bool ``True`` becomes a bare flag (``--quiet``);
    ``False`` / ``None`` is dropped.
    """

    overrides: dict[str, Any] = field(default_factory=dict)
    label: str | None = None  # optional human-readable tag for logs

    def to_eval_argv(self) -> list[str]:
        argv: list[str] = []
        for key, value in self.overrides.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                argv.append(flag)
            elif value is False or value is None:
                continue
            elif isinstance(value, list):
                argv += [flag, ",".join(str(v) for v in value)]
            else:
                argv += [flag, str(value)]
        return argv

    def describe(self) -> str:
        if self.label:
            return self.label
        bits: list[str] = []
        for k in ("backend", "synthetic_dialect", "limit", "questions"):
            if k in self.overrides and self.overrides[k] is not None:
                bits.append(f"{k}={self.overrides[k]}")
        return ", ".join(bits) or "default"


# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------
def build_vllm_command(
    *,
    base_model: str,
    lora_path: Path | None,
    lora_name: str,
    port: int,
    host: str,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    dtype: str | None,
    extra_serve_args: list[str],
) -> list[str]:
    """Construct the ``vllm serve`` argv.

    We use the modern ``vllm serve <model>`` entrypoint (vllm >= 0.4)
    rather than the older ``python -m vllm.entrypoints.openai.api_server``
    form. When ``lora_path`` is given, ``--enable-lora`` +
    ``--lora-modules <name>=<path>`` registers the adapter under
    ``<name>``, which is the model id clients send. When ``lora_path``
    is ``None`` we serve the base model bare (no LoRA flags) for
    baseline evals; clients then send ``base_model`` as the model id.
    """
    cmd = [
        "vllm",
        "serve",
        base_model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if lora_path is not None:
        cmd += [
            "--enable-lora",
            "--lora-modules",
            f"{lora_name}={lora_path}",
        ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if dtype:
        cmd += ["--dtype", dtype]
    cmd += extra_serve_args
    return cmd


def wait_for_health(
    base_url: str,
    *,
    timeout_s: float,
    interval_s: float,
    proc: subprocess.Popen[bytes] | None = None,
) -> None:
    """Poll ``<base_url>/models`` until it returns 200 or we time out.

    Uses ``/v1/models`` rather than ``/health`` because the LoRA only
    becomes available once the engine has finished loading and the
    adapter has been registered, which is what we actually care about.
    Aborts early if the server subprocess crashes.
    """
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    with httpx.Client(timeout=5.0) as c:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"vllm server exited prematurely with code {proc.returncode}; "
                    "check logs above for the failure"
                )
            try:
                r = c.get(f"{base_url.rstrip('/')}/models")
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as exc:
                last_err = str(exc)
            time.sleep(interval_s)
    raise TimeoutError(
        f"vllm server did not become healthy within {timeout_s:.0f}s "
        f"(last error: {last_err})"
    )


def shutdown_server(proc: subprocess.Popen[bytes], *, grace_s: float = 10.0) -> None:
    """SIGTERM the vllm process, escalate to SIGKILL after ``grace_s``."""
    if proc.poll() is not None:
        return
    print(f"[serve_lora] stopping vllm server (pid {proc.pid})", file=sys.stderr)
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    print("[serve_lora] vllm did not exit on SIGTERM; sending SIGKILL", file=sys.stderr)
    try:
        proc.kill()
        proc.wait(timeout=grace_s)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


# ---------------------------------------------------------------------------
# Eval dispatch
# ---------------------------------------------------------------------------
def run_one_eval(
    *,
    base_url: str,
    model_id: str,
    config: RunConfig,
    passthrough: list[str],
    default_prompt_mode: str | None = None,
) -> int:
    """Run ``eval.__main__.main`` once with the given config.

    Calls in-process (rather than spawning a subprocess) because the
    eval module is pure-Python + httpx; importing it is cheap and we
    avoid double-loading the venv. Each call gets a fresh ``argv``.

    ``model_id`` is what we send as ``--model`` to eval; it's the LoRA
    module name when serving an adapter, or the base model id when
    running the base model bare.

    ``default_prompt_mode`` is folded into the config's overrides only
    if neither the config nor the trailing passthrough already
    specifies ``--prompt-mode``. This lets per-run JSON entries or
    trailing args win without forcing the caller to repeat the
    default everywhere.
    """
    from eval.__main__ import main as eval_main  # noqa: PLC0415

    if (
        default_prompt_mode is not None
        and "prompt_mode" not in config.overrides
        and "--prompt-mode" not in passthrough
    ):
        config = RunConfig(
            overrides={**config.overrides, "prompt_mode": default_prompt_mode},
            label=config.label,
        )

    argv = [
        "--provider",
        "vllm",
        "--vllm-base-url",
        base_url,
        "--model",
        model_id,
    ]
    argv += config.to_eval_argv()
    argv += passthrough

    print(
        f"\n[serve_lora] === eval run: {config.describe()} ===\n"
        f"[serve_lora] argv: {' '.join(shlex.quote(a) for a in argv)}",
        file=sys.stderr,
    )
    return eval_main(argv)


def expand_run_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Build the list of run configs from CLI args.

    Precedence: ``--runs`` > ``--dialects`` > single inline run.
    """
    if args.runs:
        runs_path = Path(args.runs)
        data = json.loads(runs_path.read_text())
        if not isinstance(data, list):
            raise ValueError(
                f"--runs file {runs_path} must contain a JSON list of dicts"
            )
        configs: list[RunConfig] = []
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"--runs[{i}] is not a dict: {entry!r}")
            label = entry.pop("label", None)
            configs.append(RunConfig(overrides=entry, label=label))
        return configs

    base_overrides: dict[str, Any] = {}
    for k in ("backend", "questions", "limit", "concurrency", "max_retries", "output"):
        v = getattr(args, k, None)
        if v is not None:
            base_overrides[k] = v

    if args.dialects:
        dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]
        if not dialects:
            raise ValueError("--dialects parsed to an empty list")
        # When --dialects is set, we default backend to synthetic so the
        # shortcut Just Works without also requiring --backend synthetic.
        base_overrides.setdefault("backend", "synthetic")
        configs = []
        for d in dialects:
            ov = dict(base_overrides)
            ov["synthetic_dialect"] = d
            label = f"dialect={d}"
            if "limit" in ov:
                label += f" limit={ov['limit']}"
            configs.append(RunConfig(overrides=ov, label=label))
        return configs

    return [RunConfig(overrides=base_overrides)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.serve_lora",
        description=(
            "Serve a LoRA adapter atop a base model in vLLM, then run one "
            "or more eval configs against the served endpoint. Tears the "
            "server down on exit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # one eval per dialect, 20 questions each, 4 threads\n"
            "  python -m eval.serve_lora \\\n"
            "      --lora-path outputs/grpo_qwen3_4b_sql/lora \\\n"
            "      --dialects aggressive_alien,mild_postgres_ish \\\n"
            "      --limit 20 --concurrency 4\n\n"
            "  # config file driving multiple heterogeneous runs\n"
            "  python -m eval.serve_lora --lora-path ... --runs my_runs.json\n\n"
            "  # reuse a vllm server you started yourself\n"
            "  python -m eval.serve_lora --no-server --lora-name my-lora \\\n"
            "      --backend sqlite --limit 50\n"
        ),
    )

    # ---- vLLM serve ----
    p.add_argument(
        "--base-model",
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="HF model id to load as the vLLM base (default: %(default)s).",
    )
    p.add_argument(
        "--lora-path",
        default=None,
        type=Path,
        help=(
            "Path to the LoRA adapter directory written by training. "
            "Omit to serve and evaluate the bare base model (useful as a "
            "baseline against trained checkpoints)."
        ),
    )
    p.add_argument(
        "--lora-name",
        default=None,
        help=(
            "Model id to register the LoRA under (and pass to eval as --model). "
            "Default: last path component of --lora-path. Ignored when no "
            "--lora-path is given."
        ),
    )
    p.add_argument("--host", default="0.0.0.0", help="vLLM bind host (default: %(default)s).")
    p.add_argument("--port", type=int, default=8000, help="vLLM port (default: %(default)s).")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM --gpu-memory-utilization (default: %(default)s).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM --max-model-len. Pass 0 to omit and let vllm pick (default: %(default)s).",
    )
    p.add_argument(
        "--dtype",
        default=None,
        help="vLLM --dtype passthrough (e.g. bfloat16). Default: vllm auto.",
    )
    p.add_argument(
        "--vllm-extra",
        default="",
        help=(
            "Extra args appended to the `vllm serve` invocation, parsed with "
            "shlex (e.g. --vllm-extra='--enforce-eager --max-num-seqs 32')."
        ),
    )
    p.add_argument(
        "--server-startup-timeout",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT_S,
        help="Seconds to wait for /v1/models to return 200 (default: %(default)s).",
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        help=(
            "Skip launching vllm; assume a compatible server is already "
            "running at --vllm-base-url."
        ),
    )
    p.add_argument(
        "--vllm-base-url",
        default=None,
        help=(
            "Override the OpenAI-compatible base URL. Default derived "
            "from --host/--port."
        ),
    )
    p.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave the vllm server running after all evals finish.",
    )

    # ---- run configs ----
    p.add_argument(
        "--runs",
        default=None,
        help=(
            "Path to a JSON file with a list of run-config dicts. Each entry's "
            "keys become --<flag> overrides forwarded to `python -m eval`."
        ),
    )
    p.add_argument(
        "--dialects",
        default=None,
        help=(
            "Comma-separated dialect ids; runs one eval per dialect with "
            "--backend synthetic --synthetic-dialect <d>. Mutually exclusive "
            "with --runs."
        ),
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Forwarded to eval --backend (default: eval's own default = sqlite).",
    )
    p.add_argument("--questions", default=None, help="Forwarded to eval --questions.")
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Forwarded to eval --limit.",
    )
    p.add_argument(
        "--concurrency",
        "-j",
        type=int,
        default=None,
        help="Forwarded to eval --concurrency.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Forwarded to eval --max-retries.",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Forwarded to eval --output. Note this is a single file path so "
            "only meaningful for single-run invocations; for multi-run mode "
            "let eval pick its default per-run path."
        ),
    )

    p.add_argument(
        "--prompt-mode",
        choices=["plain", "tag"],
        default="tag",
        help=(
            "System-prompt format passed to each eval run. Defaults to 'tag' "
            "because this wrapper's primary use case is evaluating LoRAs "
            "from train/grpo_sql.py, which trains the model to emit "
            "<SQL>...</SQL>. Use 'plain' if you're benchmarking a base "
            "model that wasn't trained on the tag protocol. Per-run "
            "entries in --runs JSON can override this with their own "
            "'prompt_mode' key."
        ),
    )

    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one eval run fails, log it and proceed to the next run.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Split off trailing passthrough args after a literal "--".
    passthrough: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        passthrough = argv[idx + 1 :]
        argv = argv[:idx]

    args = _build_parser().parse_args(argv)

    if args.runs and args.dialects:
        print(
            "error: --runs and --dialects are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    if (
        not args.no_server
        and args.lora_path is not None
        and not args.lora_path.exists()
    ):
        print(
            f"error: --lora-path {args.lora_path} does not exist",
            file=sys.stderr,
        )
        return 2

    # When no LoRA is given, the model id we send to eval is the bare base
    # model. When a LoRA *is* given, the LoRA module name (registered via
    # --lora-modules <name>=<path>) is what clients must send.
    if args.lora_path is not None:
        if args.lora_name:
            model_id = args.lora_name
        else:
            # Auto-derive a unique-ish id. The conventional layout is
            # ``<run_dir>/lora`` (final adapter) or
            # ``<run_dir>/checkpoint-NNN`` (intermediate), so two runs
            # would otherwise collide on ``lora`` and overwrite each
            # other's eval result file. Walk up to the run dir when
            # the leaf name is generic.
            path = args.lora_path.resolve()
            if path.name in {"lora", "adapter"} or path.name.startswith("checkpoint-"):
                model_id = f"{path.parent.name}_{path.name}"
            else:
                model_id = path.name
    else:
        if args.lora_name:
            print(
                "warning: --lora-name has no effect without --lora-path; "
                "using --base-model as the eval model id",
                file=sys.stderr,
            )
        model_id = args.base_model

    base_url = args.vllm_base_url or f"http://{args.host or 'localhost'}:{args.port}/v1"
    # Localhost is friendlier than 0.0.0.0 for httpx; only the *bind* host
    # needs to be 0.0.0.0 for external access.
    if base_url.startswith("http://0.0.0.0"):
        base_url = base_url.replace("0.0.0.0", "127.0.0.1", 1)

    try:
        run_configs = expand_run_configs(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: failed to build run configs: {exc}", file=sys.stderr)
        return 2

    lora_desc = str(args.lora_path) if args.lora_path is not None else "<none: base model>"
    print(
        f"[serve_lora] base_model={args.base_model} lora={lora_desc} "
        f"model_id={model_id} base_url={base_url}",
        file=sys.stderr,
    )
    print(
        f"[serve_lora] {len(run_configs)} eval run(s) queued:",
        file=sys.stderr,
    )
    for i, rc in enumerate(run_configs):
        print(f"  [{i+1}] {rc.describe()}", file=sys.stderr)

    proc: subprocess.Popen[bytes] | None = None
    if not args.no_server:
        extra_serve_args = shlex.split(args.vllm_extra) if args.vllm_extra else []
        max_model_len = args.max_model_len if args.max_model_len > 0 else None
        cmd = build_vllm_command(
            base_model=args.base_model,
            lora_path=args.lora_path.resolve() if args.lora_path is not None else None,
            lora_name=model_id,
            port=args.port,
            host=args.host,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            extra_serve_args=extra_serve_args,
        )
        print(
            f"[serve_lora] launching: {' '.join(shlex.quote(a) for a in cmd)}",
            file=sys.stderr,
        )
        # start_new_session=True puts vllm in its own pgid so a Ctrl-C in
        # this process tree doesn't double-signal it (we'll SIGTERM it
        # explicitly in the finally block).
        proc = subprocess.Popen(cmd, start_new_session=True)

        # Forward our own SIGTERM/SIGINT to the cleanup path even if we're
        # mid-eval. Default behavior is fine (KeyboardInterrupt percolates
        # up) but on SIGTERM Python would exit without running finally
        # unless we install a handler.
        def _term_handler(signum: int, _frame: Any) -> None:
            print(
                f"[serve_lora] received signal {signum}; tearing down",
                file=sys.stderr,
            )
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _term_handler)

    exit_code = 0
    try:
        if proc is not None:
            print(
                f"[serve_lora] waiting for {base_url}/models (timeout "
                f"{args.server_startup_timeout:.0f}s)...",
                file=sys.stderr,
            )
            wait_for_health(
                base_url,
                timeout_s=args.server_startup_timeout,
                interval_s=DEFAULT_HEALTH_INTERVAL_S,
                proc=proc,
            )
            print("[serve_lora] vllm is healthy", file=sys.stderr)

        for i, rc in enumerate(run_configs):
            try:
                rc_code = run_one_eval(
                    base_url=base_url,
                    model_id=model_id,
                    config=rc,
                    passthrough=passthrough,
                    default_prompt_mode=args.prompt_mode,
                )
            except Exception as exc:  # noqa: BLE001 - we want to keep going
                print(
                    f"[serve_lora] run {i+1}/{len(run_configs)} crashed: {exc}",
                    file=sys.stderr,
                )
                rc_code = 1
            if rc_code != 0:
                print(
                    f"[serve_lora] run {i+1}/{len(run_configs)} "
                    f"({rc.describe()}) exited {rc_code}",
                    file=sys.stderr,
                )
                exit_code = rc_code
                if not args.continue_on_error:
                    break
            else:
                print(
                    f"[serve_lora] run {i+1}/{len(run_configs)} "
                    f"({rc.describe()}) ok",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("[serve_lora] interrupted", file=sys.stderr)
        exit_code = 130
    except (RuntimeError, TimeoutError) as exc:
        print(f"[serve_lora] {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if proc is not None:
            if args.keep_server and exit_code == 0:
                print(
                    f"[serve_lora] --keep-server set; vllm still running at "
                    f"{base_url} (pid {proc.pid}). Stop it with: kill {proc.pid}",
                    file=sys.stderr,
                )
            else:
                shutdown_server(proc)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
