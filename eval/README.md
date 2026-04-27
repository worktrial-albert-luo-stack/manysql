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

# Evaluating a LoRA trained by `train/grpo_sql.py`: switch the prompt
# format so the system message matches what the model was trained on.
uv run manysql-eval --provider vllm \
    --vllm-base-url http://localhost:8000/v1 \
    --model my-lora \
    --prompt-mode tag --backend synthetic \
    --synthetic-dialect aggressive_alien
```

### Prompt formats (`--prompt-mode`)

Two prompt formats ship; the dialect hints, schema body, and reference
SQL all stay identical between them:

| Mode | Output instruction | Use when |
| --- | --- | --- |
| `plain` (default) | "Return ONLY the SQL query, with no markdown, no fences, no commentary." | Closed-source frontier models (GPT-4o, Claude, Gemini) — they follow plain-text instructions reliably. |
| `tag` | "Wrap the final SQL between `<SQL>` and `</SQL>` tags." | Any LoRA produced by `train/grpo_sql.py` — the GRPO reward function trains the model to emit `<SQL>...</SQL>`, and a plain-mode eval would instruct the model to do something other than what it's been trained to do. |

`extract_sql` is tag-aware in *either* mode: a tag-trained model run
under `--prompt-mode plain` is still scored correctly (the extractor
strips the tags). The mode mostly affects whether the system prompt
*contradicts* what the model learned.

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
├── perf_bench.py          # Postgres vs manysql `postgres_clone` perf bench
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

## Performance benchmarking (Postgres vs `postgres_clone`)

The LLM-correctness harness above is orthogonal to *engine* perf. To compare
how fast each engine actually executes the same query, `eval/perf_bench.py`
runs a curated dialect-neutral suite against:

1. A real **PostgreSQL** server (via `psycopg`).
2. The **manysql `postgres_clone`** dialect (Lark parse → IR lowering →
   Polars execution), generated from
   [`manysql/spec/examples/postgres_clone.py`](../manysql/spec/examples/postgres_clone.py).

Both engines see the same `github_events` rows (the deterministic synthetic
dataset from `eval/dataset/github_events.py`), the same 15 SQL statements,
and the same warmup/repeat schedule, so the resulting timings are directly
comparable.

### Setup

```bash
# 1. Install the bench extra (adds psycopg[binary]).
uv sync --extra dev --extra bench

# 2. Generate the postgres_clone dialect package (one-time, deterministic).
uv run manysql-codegen gen postgres_clone

# 3. Point the bench at a Postgres instance. The fastest path is Docker:
docker run -d --name manysql-pg \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=manysql_bench \
    postgres:16
export BENCH_POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/manysql_bench
```

`--postgres-url`, `BENCH_POSTGRES_URL`, and `DATABASE_URL` are all
recognized (in that priority order). The bench creates / truncates the
`github_events` table on each run, so it's safe to point at a throwaway
DB but **not** at production.

### Running the bench

```bash
# Show the curated query suite (no DB needed).
uv run manysql-perf-bench --list

# Default: 50k rows, 1 warmup + 5 timed reps per query.
uv run manysql-perf-bench

# Larger dataset, more reps:
uv run manysql-perf-bench --rows 200000 --repeats 10 --warmup 2

# Subset of queries:
uv run manysql-perf-bench --queries q01_count_stars,q05_top_pushers_having

# Persist per-query timings as JSON for plotting / regression tracking:
uv run manysql-perf-bench --output bench_$(date +%s).json

# Skip the result-equivalence check (faster, but won't catch divergences).
uv run manysql-perf-bench --no-verify
```

The script prints a Rich summary table with median / p95 latency for each
engine and the manysql/Postgres ratio, plus a one-line geometric-mean
speedup at the end. See [`PERFORMANCE.md`](PERFORMANCE.md) for an
annotated set of recent results (200k rows, 15/15 row-equivalence,
geomean ~1.03x), including which query shapes each engine wins on
and why. Equivalence checking normalizes Postgres `Decimal`
to `float`, rounds floats to 6 dp, and compares unordered when the SQL
has no `ORDER BY`, so minor numeric / ordering differences don't trip
false negatives — but real semantic divergences (e.g. NULL ordering,
case folding) will surface as a `[mismatch]` annotation.

### What's *not* in the suite (and why)

* `DATE_PART('year', CAST(text_col AS TIMESTAMP))` — Postgres handles the
  string→timestamp cast natively, but manysql's Cast currently delegates
  to Polars's strict=False cast, which returns NULL on ISO 8601 strings
  (Polars expects `str.to_datetime(format=...)`). Adding the query would
  produce a real divergence rather than a perf signal. The suite covers
  the same intent via `SUBSTRING(created_at, 1, 4)` (q04).
* SQLite-only functions (`strftime`, `LOG10`, …) — not portable to
  Postgres. The LLM-correctness harness in this directory still uses
  them via the SQLite reference executor; the perf bench deliberately
  doesn't.

To add a query, append a `BenchQuery(name, description, sql)` to
`_BENCH_QUERIES` in `eval/perf_bench.py`. The SQL must parse and execute
on both engines without translation — no dialect-specific shims.

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

The wrapper defaults to `--prompt-mode tag` because its primary use case
is evaluating GRPO-trained LoRAs (which expect the `<SQL>...</SQL>`
protocol). Pass `--prompt-mode plain` to baseline a model that wasn't
trained on the tag format, or set `prompt_mode` per-entry inside a
`--runs` JSON file. Omitting `--lora-path` switches both the served
model and the eval `--model` arg to `--base-model`, which is the
canonical way to establish a no-LoRA baseline against the same dialect
configs.

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
