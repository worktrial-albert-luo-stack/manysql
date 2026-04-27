# manysql eval

Pluggable LLM SQL-generation eval harness, modeled after
[tinybirdco/llm-benchmark](https://github.com/tinybirdco/llm-benchmark).
The same NL-question pipeline (generate → execute → retry on error → score
against a reference) but with three swappable axes:

| Axis | Options |
| --- | --- |
| **LLM provider** | `openai`, `openrouter`, `vllm` (local OpenAI-compatible) |
| **Execution backend** | `sqlite` (default), `tinybird`, `synthetic` (manysql-generated dialects) |
| **Question suite** | full 50-question port of `tinybirdco/llm-benchmark` (extensible) |

The default backend is **SQLite** seeded with a small synthetic GitHub-events
corpus, so the bench runs end-to-end on a laptop with no external services.
Tinybird is supported for parity with the upstream benchmark. The
`synthetic` backend loads a manysql-generated dialect package (see
[`manysql/dialects/`](../manysql/dialects/) and the
[Synthetic dialects](#synthetic-dialects) section below) and judges the
LLM's SQL through that dialect engine.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env  # add OPENROUTER_API_KEY or OPENAI_API_KEY

# Sanity-check the seed dataset (no LLM call):
uv run manysql-eval --dry-run

# Real run with OpenRouter:
uv run manysql-eval --model anthropic/claude-sonnet-4

# OpenAI:
uv run manysql-eval --provider openai --model gpt-4o-mini

# Local vLLM serve:
uv run manysql-eval --provider vllm \
    --vllm-base-url http://localhost:8000/v1 \
    --model unsloth/Qwen3-4B-Instruct-2507

# Subset of questions:
uv run manysql-eval --model anthropic/claude-sonnet-4 \
    --questions q01_count_stars,q05_top_repos_by_year_since_2015

# Cap how many questions to run (e.g. quick smoke test on a new model):
uv run manysql-eval --provider openai --model gpt-4o-mini --limit 5
# (or the short form: -n 5)

# Parallelize LLM calls across 8 worker threads (~Nx wall-clock speedup
# until you hit the provider's rate limit / your local vLLM throughput):
uv run manysql-eval --provider openai --model gpt-4o-mini -j 8
```

The `--limit N` flag is applied *after* `--questions`, so you can combine
them (e.g. `--questions q40,q47,q50 --limit 2` runs the first two of those
three).

`--concurrency N` / `-j N` (default `1` = sequential) fans questions out
over a thread pool. Each question is independent, so threads give
near-linear wall-clock speedup on I/O-bound LLM calls. Per-question
results stay in input order in the JSON output regardless of completion
order; the in-progress lines stream in completion order so you see the
fast questions first. Bump it conservatively at first — paid providers
will start throwing 429s above their per-minute token / request budget.

`manysql-eval`, the alias `eval`, and `python -m eval` are all equivalent
entry points; pick whichever reads best for your workflow. Avoid bare
`eval` in shell scripts since `eval` is also a shell builtin.

Results land in `results/<provider>_<model>_<backend>.json`.

## Layout

```
eval/
├── __main__.py            # CLI
├── llm.py                 # OpenAI-compatible client (provider-agnostic)
├── prompt.py              # dialect-aware system prompt
├── runner.py              # generate → execute → retry → score
├── validator.py           # Jaccard / RMSE / F-score (port of tinybird's TS)
├── executors/
│   ├── base.py            # SqlExecutor protocol
│   ├── sqlite_executor.py # default
│   ├── tinybird_executor.py
│   └── synthetic_executor.py  # parses + lowers via a manysql dialect
└── dataset/
    ├── github_events.py   # SQLite schema + deterministic seeder
    └── questions.py       # NL question + reference SQL per dialect
```

## Adding a question

Append a `Question` to `eval/dataset/questions.py`. The `reference_sql` dict
is keyed by dialect substring (`'sqlite'`, `'clickhouse'`, ...) and looked
up against the executor's `dialect_label()`.

## Adding a backend

Implement `eval/executors/base.SqlExecutor` and register it in
`eval/executors/factory.py`. It needs `setup()`, `execute(sql) -> ExecResult`,
`schema_prompt()`, and `dialect_label()`.

## Adding an LLM provider

The OpenAI chat-completions wire format covers OpenAI, OpenRouter, vLLM,
Together, Groq, llama.cpp's server, etc. — point `LLMClient(base_url=...)`
at any compliant endpoint. For non-OpenAI-compatible APIs (Anthropic native,
Bedrock, …), subclass `LLMClient.chat`.

## Synthetic dialects

manysql ships a hand-written `_reference` dialect plus a codegen pipeline
that emits new dialects from a `DialectSpec`. To bench an LLM against a
generated dialect:

```bash
# 1. List the bundled spec examples.
uv run manysql-codegen --list

# 2. Generate the dialect package (deterministic emitter; near-instant).
uv run manysql-codegen gen mild_postgres_ish

# 3. Confirm the dialect engine can run the (SQLite-flavored) reference
#    SQL on the seed dataset.
uv run manysql-eval --backend synthetic \
    --synthetic-dialect mild_postgres_ish \
    --limit 5 --dry-run

# 4. Live eval: LLM SQL is parsed + executed by the dialect engine,
#    ground truth is computed via SQLite (auto-attached).
uv run manysql-eval --backend synthetic \
    --synthetic-dialect mild_postgres_ish \
    --provider openai --model gpt-4o-mini --limit 10 -j 4
```

The CLI auto-attaches a SQLite executor as the *reference executor* when
`--backend synthetic` is set, since the question suite's reference SQL
is currently SQLite-flavored. Pass `--no-reference-executor` to override
(rare; useful only after you've authored dialect-specific reference SQL).

Generated dialects live under `manysql/dialects/<name>/` and are managed
by `manysql.dialects.registry.DialectRegistry`. Each package contains
`grammar.lark`, `lowering.py`, `semantics.json`, `overrides.py`,
`metadata.json`, and `spec.json`.

To use an LLM to refine the grammar / lowering instead of the
deterministic templates:

```bash
uv run manysql-codegen gen aggressive_alien --use-llm --overwrite
```

This needs `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (configurable via
`OPENAI_MODEL` / `OPENROUTER_MODEL`).

## Serving a trained LoRA adapter

After `train/grpo_sql.py` writes a LoRA checkpoint to e.g.
`outputs/grpo_qwen3_4b_sql/lora/`, evaluating it usually means running
several configs (different dialects, different question subsets,
different backends) against the same checkpoint. Spinning up vLLM
costs ~30-60s on H100 so doing one server start per `python -m eval`
invocation is wasteful.

`eval/serve_lora.py` (alias `manysql-serve-eval`) wraps the lifecycle:
it spawns `vllm serve <base> --enable-lora --lora-modules <name>=<path>`,
waits for `/v1/models` to come up, runs each requested eval config
against `http://localhost:<port>/v1` with `--provider vllm --model <name>`,
then tears the server down.

```bash
# one eval per dialect, 20 questions each, 4 threads
uv run manysql-serve-eval \
    --lora-path outputs/grpo_qwen3_4b_sql/lora \
    --dialects aggressive_alien,mild_postgres_ish,tsql_ish \
    --limit 20 --concurrency 4

# baseline: same configs against the bare base model (no LoRA).
# Omit --lora-path; we drop --enable-lora and dispatch eval with
# --model <base-model>.
uv run manysql-serve-eval \
    --base-model unsloth/Qwen3-4B-Instruct-2507 \
    --dialects aggressive_alien,mild_postgres_ish,tsql_ish \
    --limit 20 --concurrency 4

# heterogeneous runs from a JSON config
cat > my_runs.json <<'EOF'
[
  {"backend": "sqlite", "limit": 50, "concurrency": 4},
  {"backend": "synthetic", "synthetic_dialect": "aggressive_alien",
   "questions": "q01_count_stars,q05_top_repos_by_year_since_2015"}
]
EOF
uv run manysql-serve-eval --lora-path outputs/.../lora --runs my_runs.json

# reuse a vllm server you started yourself in a tmux pane
uv run manysql-serve-eval --no-server --lora-name my-lora \
    --backend sqlite --limit 50

# trailing args after `--` forward to every eval invocation
uv run manysql-serve-eval --lora-path ... --dialects a,b -- \
    --temperature 0.2 --max-tokens 4096
```

`--keep-server` leaves vLLM running after all evals finish (handy for
ad-hoc curl probes); otherwise the server is SIGTERM'd on exit
(KeyboardInterrupt, eval failure, or successful completion).

## Roadmap

* Persist results in a structured table (DuckDB?) and add a Rich-rendered
  leaderboard view, matching the upstream `validation-results.json`.
* Add per-question `clickhouse` reference SQL alongside the existing
  `sqlite` entries so the `tinybird` backend has a non-stub reference
  rather than reusing the SQLite text.
* Author per-dialect reference SQL once the divergence-level=alien
  surface diverges enough that SQLite reference text no longer parses.
