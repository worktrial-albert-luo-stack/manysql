# Engine perf: Postgres vs `postgres_clone`

Results from running [`eval/perf_bench.py`](perf_bench.py) — a curated,
dialect-neutral SQL suite executed against:

1. **PostgreSQL 16** via `psycopg` (real Postgres binary, default config).
2. **manysql `postgres_clone`** — Lark parse → IR lowering → Polars
   execution, using the dialect package generated from
   [`manysql/spec/examples/postgres_clone.py`](../manysql/spec/examples/postgres_clone.py).

Both engines see the same `github_events` synthetic dataset
(`eval/dataset/github_events.py`), the same 15 SQL statements, the same
btree indexes (`event_type`, `repo_name`, `actor_login`, `created_at`),
and the same warmup / repeat schedule. Equivalence is checked per
query (Decimal→float normalization, 6-dp float rounding, set-compare
when there's no `ORDER BY`).

How to reproduce all numbers below is at the bottom.

## Headline

| dataset | repeats | geomean (manysql / pg) | row-equivalence |
| --- | --- | --- | --- |
| 50k rows  | 1 warmup + 5 timed | **3.76x** (Postgres faster on average) | 15/15 |
| 200k rows | 2 warmup + 10 timed | **1.03x** (parity) | 15/15 |

The crossover between 50k and 200k tells the story. manysql carries a
fixed per-query overhead (Lark parse → IR lower → Polars plan build)
that dominates on small datasets, but its vectorized Polars execution
scales better than Postgres's row-based engine, so by 200k rows the two
engines are essentially tied in aggregate — with each winning on
different query shapes.

## Per-query results at 200k rows

| query | pg p50 (ms) | manysql p50 (ms) | manysql/pg | rows | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| q01_count_stars              |   3.87 |   4.39 | 1.1x |     1 | pg, narrowly |
| q02_top_starred_repos        |   6.79 |  10.67 | 1.6x |    10 | pg |
| q03_count_distinct_repos     |  33.31 |   4.57 | **0.1x** |     1 | **manysql ~7x** |
| q04_stars_by_year            |   9.31 |  12.20 | 1.3x |    11 | pg |
| q05_top_pushers_having       |   8.28 |  12.90 | 1.6x |    20 | pg |
| q06_filter_like              |   9.77 |   7.93 | 0.8x |     1 | manysql |
| q07_case_classification      |  10.08 |  14.36 | 1.4x |     3 | pg |
| q08_in_subquery              |   4.68 |  19.10 | **4.1x** |    10 | **pg** |
| q09_window_row_number        |  17.12 |  14.63 | 0.9x |    10 | manysql |
| q10_cte_aggregate            |   6.62 |  11.64 | 1.8x |     1 | pg |
| q11_union_distinct           |  15.34 |   8.94 | 0.6x |    20 | manysql |
| q12_sum_arith                |  10.54 |  15.81 | 1.5x |    10 | pg |
| q13_month_buckets            |   3.53 |  11.28 | **3.2x** |   132 | **pg** |
| q14_distinct_pairs           |   9.77 |  12.89 | 1.3x |    22 | pg |
| q15_distinct_repos_per_actor |  76.64 |   8.55 | **0.1x** |    20 | **manysql ~9x** |

Identical row sets on **15/15** queries (after the
result-equivalence normalization above).

## What jumps out

### Where manysql wins big

`count(DISTINCT)` and high-cardinality `GROUP BY`:

* **q03** (`SELECT count(DISTINCT repo_name)`): manysql ~7x faster
  (33ms pg vs 4.6ms manysql). Polars's hash-distinct over a single
  Utf8 column is tight; Postgres has to hash-aggregate row-by-row.
* **q15** (top 20 actors by distinct repos touched): manysql ~9x
  faster (77ms pg vs 8.5ms manysql). Same shape, made harder for
  Postgres because the per-actor distinct count is computed against a
  much larger group-key cardinality (~50k actors after dedup).
  Postgres's `idx_pg_repo_name` doesn't help here — the GROUP BY is
  on `actor_login`.
* **q11** (`UNION DISTINCT` of two scans): manysql ~1.7x faster.
  Polars's `concat` + `unique` is cheaper than Postgres's
  HashAggregate(SetOp).
* **q06** / **q09**: small wins on `LIKE` filtering and `ROW_NUMBER`
  windowing — Polars vectorization eats fixed overhead at this size.

### Where Postgres wins big

Subquery materialization and string-prefix bucketing:

* **q08** (`IN (SELECT ...)` of tensorflow PR authors): pg 4.1x
  faster. Postgres's planner turns the `IN`-subquery into a hash
  semi-join cleanly. manysql's lowering currently materializes the
  subquery as an independent plan and joins via Polars membership,
  paying double-vectorization overhead for what should be a single
  hash-build.
* **q13** (`SUBSTRING(created_at, 1, 7) AS yyyymm` GROUP BY): pg
  3.2x faster. Postgres folds the constant-length substring into a
  fast text op directly inside the aggregator. manysql goes through
  a generic Polars `str.slice` followed by a separate `group_by`,
  which is heavier per row.
* **q01** / **q05** / **q10** / **q12**: small wins on cheap
  aggregations and CTE/sum-arithmetic. The fixed-cost overhead of
  manysql's parse-and-lower path is roughly 4–8 ms per query at 200k
  rows, which is a real fraction of these queries' total budget.

### Where it's a wash

q02, q04, q07, q14 — all in the 1.3–1.6x manysql/pg range. These are
"normal" GROUP BY + ORDER BY shapes with mid cardinality; the engines
take comparable time and the numbers will move around 20–30% across
runs.

## Caveats

* **Embedded Postgres**, not a tuned production server. The bench uses
  [`pgserver`](https://pypi.org/project/pgserver/) (real Postgres 16
  binary, in-process unix socket, default `shared_buffers`,
  `work_mem`, `max_parallel_workers_per_gather=2` — but very few of
  these plans qualify for parallel scan at 200k rows). A tuned
  production Postgres on the same hardware would close some of the
  manysql wins on q03/q11/q15 (more work_mem helps hash-distinct);
  it wouldn't change the q08/q13 picture much.
* **Cold-start excluded.** Both engines run a warmup pass that's
  thrown away. Postgres bootstrap (~3s for `pgserver` to spin up)
  and manysql dialect load (~1.2s for the parser + Polars frame at
  200k rows) are reported once at setup time, not folded into
  per-query timings.
* **Single machine, single thread per query.** Polars defaults to its
  thread pool, but at this size most queries finish in <20ms, where
  parallelism overhead can hurt. Postgres's per-query parallelism is
  effectively off for these plans.
* **Indexes match.** Both engines have btree indexes on `event_type`,
  `repo_name`, `actor_login`, `created_at`. There's no scenario where
  one engine has an index advantage the other doesn't.
* **Row-set equivalence ≠ semantic equivalence.** The bench checks
  `set(rows)` (or ordered list when `ORDER BY` is present) match. A
  divergence in NULL ordering, case folding, or division-by-zero
  behavior on a query *that doesn't exercise it* would still pass.
  The dialect's full semantic profile is exercised separately by the
  cross-dialect golden suite (`manysql-dialect diff postgres_clone`).

## Reproducing

The bench is fully deterministic — same seed, same dataset, same
query set:

```bash
# 1. Install the bench extra (adds psycopg[binary]).
uv sync --extra dev --extra bench

# 2. Generate the postgres_clone dialect package.
uv run manysql-codegen gen postgres_clone

# 3a. Easiest: ephemeral embedded Postgres via pgserver.
uv pip install pgserver
uv run python - <<'PY'
import os, tempfile, pgserver, subprocess, sys
srv = pgserver.get_server(tempfile.mkdtemp(prefix="pgserver_bench_"))
env = {**os.environ, "BENCH_POSTGRES_URL": srv.get_uri()}
sys.exit(subprocess.run([
    "uv", "run", "manysql-perf-bench",
    "--rows", "200000", "--warmup", "2", "--repeats", "10",
    "--output", "results/perf_postgres_clone_200k.json",
], env=env).returncode)
PY

# 3b. Or against your own Postgres (Docker, RDS, local install, ...):
export BENCH_POSTGRES_URL=postgresql://user:pass@host:5432/dbname
uv run manysql-perf-bench --rows 200000 --warmup 2 --repeats 10
```

Per-query timings land in `results/perf_postgres_clone_200k.json`
(or whichever path you pass to `--output`) for plotting / regression
tracking.

## Hardware

Numbers above were collected on:

* Apple M4 Pro, macOS Darwin 25.4.0, 16-core CPU
* Python 3.13 via `uv`
* Postgres 16 (pgserver-bundled binary)
* `polars` 1.x (whatever the lockfile pins; see `uv.lock`)

A different machine class will shift absolute numbers but should
preserve the per-query winner pattern, since the per-shape bottleneck
(hash-distinct vs subquery-semi-join vs string-slice) is a property
of the planner and execution model, not the hardware.
