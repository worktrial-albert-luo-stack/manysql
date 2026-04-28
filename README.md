# manysql

A generator of synthetic **SQL dialects** with full per-dialect query
engines, plus the verification, evaluation, and RL infrastructure to
use them as LLM training data.

manysql is **scope-locked to SQL** (Tier-A: relational algebra over
batch, single-source, read-only data). Generating non-SQL query
languages — Cypher, jq, KQL, PRQL, streaming, procedural — is
explicitly out of scope; see [`manysql/ir/SCOPE.md`](manysql/ir/SCOPE.md).

The full design walkthrough — SQL properties → architecture options →
pipeline / engine internals → training & evaluation results → future
experiments — lives in [`PRESENTATION.md`](PRESENTATION.md). This README
is the short version: setup, design overview, headline results.

## Setup

```bash
uv sync --extra dev
cp .env.example .env  # add OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY as needed
```

## End-to-end smoke test

```bash
# 1. Generate a dialect package (deterministic, ~instant).
uv run manysql-codegen gen mild_postgres_ish

# 2. Inspect what changed vs. the reference surface.
uv run manysql-dialect diff mild_postgres_ish

# 3. Run a real eval. Reference SQL is computed via the auto-attached
#    SQLite executor; the candidate dialect engine parses + executes
#    the LLM's reply.
uv run manysql-eval --backend synthetic \
    --synthetic-dialect mild_postgres_ish \
    --provider openai --model gpt-4o-mini \
    --limit 10 -j 4

# 4. Aggressive surface where the deterministic emitter alone won't
#    suffice — force the LLM refine loop:
uv run manysql-codegen gen aggressive_alien --use-llm --overwrite

# 5. Populate a benchmark with many synthetic dialects in one shot.
uv run manysql-codegen batch --n 5 \
    --prior "variants between mssql and snowflake" \
    --theme mixed \
    --inspired-by sql_server,snowflake

# 6. Run the test suite (oracles, harness, golden queries, codegen
#    pipeline, registry, property fuzzing, cross-dialect differential).
uv run pytest -n auto
```

CLI entry points (installed by `uv sync`):

| Command | Purpose |
| --- | --- |
| `manysql-codegen gen <spec>` | Materialize a dialect package from a `DialectSpec`. `--use-llm` runs at least one LLM refinement pass on top of the deterministic baseline. |
| `manysql-codegen batch --n N --prior <vibe>` | Outer-loop campaign: design N diverse specs, fan them through the codegen pipeline. Manifest at `manysql/dialects/_campaigns/<id>.json`. |
| `manysql-dialect diff <name>` | Side-by-side diff of a generated dialect's reskinned battery vs. the canonical reference SQL. |
| `manysql-eval` (alias `eval`) | Pluggable LLM SQL benchmark: NL question → model SQL → execute → retry on error → score. Backends: `sqlite`, `tinybird`, `synthetic`. Providers: `openai`, `openrouter`, `vllm`. See [`eval/README.md`](eval/README.md). |

## Design overview

```
DialectSpec  ─┐
              ├─►  codegen pipeline  ─►  dialect package on disk
              │      (deterministic emitters + LLM refine loop)         │
              │                                                         │
              │                                            ┌────────────┴────────────┐
              │                                            ▼                         ▼
              │                                       parse battery            IR-equivalence
              │                                       (Lark grammar)           battery (lowered
              │                                                                Plans match the
              │                                                                reference Plans)
              │
              │   shared infrastructure (hand-written, scope-locked):
              │   ├─ logical-plan IR (relational algebra, Tier-A only)
              │   ├─ Polars/PyArrow executor parameterized by SemanticConfig
              │   ├─ multi-oracle harness (DuckDB, SQLite, ref interpreter,
              │   │  property oracle, cross-dialect differential)
              │   └─ Parquet-backed deterministic test catalog
              │
              └─►  manysql-eval benchmark + train/ RL env (GRPO over
                   synthetic dialects)
```

A generated dialect is a directory of files under
`manysql/dialects/<name>/`:

```
grammar.lark       Lark grammar — surface syntax of this dialect
lowering.py        parse tree → manysql IR Plan          (open-world)
semantics.json     SemanticConfig values               (closed-world)
overrides.py       FUNCTIONS / OPERATORS dicts          (open, optional)
passes.py          PRE_EXECUTION_PASSES list            (open, optional)
effects.py         EFFECTS dict                         (open, optional)
metadata.json      provenance, retry log, lifecycle state
spec.json          the DialectSpec this was generated from
battery.json       parse + IR-equivalence batteries
examples.sql       parse battery rendered as inspectable dialect SQL
```

### Key design decisions

The architecture is forced by five facts about SQL (full argument in
[`PRESENTATION.md` §1](PRESENTATION.md)):

| SQL property | Force on the design |
|---|---|
| **Surface ≠ semantics** — the same query, byte-for-byte, can return different rows on different engines (NULL ordering, `1/0`, `LIKE` case sensitivity, `5/2`). | Token-level transpilation isn't enough. The system must *execute* candidate SQL under the dialect's actual runtime. |
| **Layered (lex / syn / sem)** — three roughly-independent layers of variation. | Codegen parameterizes the layers independently — orthogonal knobs. |
| **Closed-world divergence catalog (~15 axes)** — identifier folding, null order, divide-by-zero, integer division, `LIKE` case sensitivity, string-concat operator, set-op default, boolean truthiness, etc. | A small enum-driven config (`SemanticConfig`) covers most divergence as data, not code branches. |
| **Long tail exists** — collations, novel functions, plan sugar (`QUALIFY`, `LIMIT … WITH TIES`). | Per-dialect Python escape hatches (`overrides.py`, `passes.py`, `effects.py`) alongside the enums. |
| **Relational algebra is the common substrate** — every Tier-A SQL query lowers to ~14 operators. | One IR + one executor parameterized by `SemanticConfig` — not N forks. |

The four extension lanes a dialect feature can land in, in priority
order:

| Rule | Where the feature lives | Example |
|---|---|---|
| **1. Pure surface** | `SurfaceSpec` knob (closed enum) | `LIMIT n` ⇄ `OFFSET n ROWS FETCH NEXT n ROWS ONLY`; `\|\|` ⇄ `+` |
| **2. Runtime divergence in a small enum** | `SemanticConfig` knob (closed enum) | NULL ordering on `ORDER BY`; divide-by-zero policy; `LIKE` case sensitivity |
| **3. Canonical IR is the right *shape*, surface produces a non-canonical marker** | `passes.py` Plan→Plan rewrite | `LIMIT N WITH TIES` desugars to `Window(rank) → Filter(rank ≤ N) → Project` |
| **4. Canonical IR is the right *shape*, executor's *implementation* differs** | `effects.py` handler at a registered name | `text_eq` / `text_neq` / `text_in_pattern` for case-insensitive collation |
| **5. Canonical IR can't represent the feature** | Tier-B IR extension via RFC | arrays, JSON path expressions, sampling |

Tier-B IR extensions (arrays/structs, JSON, regex flavor, sampling,
time travel, MERGE, pivot/unpivot) are deferred to v1.5 behind RFCs
under [`manysql/ir/rfcs/`](manysql/ir/rfcs/); each *grows* the IR
rather than redesigning it, so Tier-A dialects keep working.

### Codegen pipeline

```
DialectSpec  ─►  deterministic emitters  ─►  parse battery   ─┐
(Pydantic)         (templated from spec)                       │
                          │                                    ▼
                          ▼                       (passes? — done, ship)
                     LLM refine loop                          │
                  (only if templates                          ▼
                   can't express, or                     IR battery
                   --use-llm forced)                          │
                          │                                   ▼
                          ▼                          rejection battery
                       rollback if                            │
                       any battery                            ▼
                       regresses                  card-conformance gate
                                                              │
                                                              ▼
                                                    package on disk
```

Four validation gates: parse battery (every example parses), rejection
battery (the grammar refuses reference-form syntax the spec disallows),
IR equivalence (lowered plan matches the reference dialect's plan),
card conformance (the dialect card matches what the grammar accepts).
Any LLM iteration that regresses any battery is rolled back — the
deterministic baseline is the floor.

### Verification harness

`manysql/oracle/` runs a Plan through every applicable oracle (DuckDB,
SQLite, a hand-written reference interpreter independent of the Polars
executor, structural property invariants, and a cross-dialect
differential check) and returns `PASS / FAIL / NEEDS_REVIEW /
NO_ORACLE`. Cross-dialect disagreement is either curated
semantic-divergence training data or a codegen bug.

## Repository layout

```
manysql/
├── ir/              # logical-plan IR + SCOPE.md + RFCs/
├── executor/        # Polars/PyArrow IR executor
├── oracle/          # DuckDB / SQLite / reference interpreter / property /
│                    #   cross-dialect oracles + harness
├── storage/         # Parquet-backed deterministic test catalog
├── spec/            # SemanticConfig + DialectSpec schemas (Pydantic)
├── dialects/        # generated dialect packages, registry, _campaigns/
├── codegen/         # deterministic emitters + LLM refine agents
│                    #   + parse / IR / rejection batteries
├── golden/          # hand-curated SQL corpus (queries.py)
├── llm/             # thin OpenAI/OpenRouter/Anthropic chat client
└── verify/          # harness loops over goldens

eval/                # pluggable LLM SQL benchmark + perf bench
                     #   (see eval/README.md, eval/PERFORMANCE.md)
train/               # GRPO/RLVR training entrypoints + SQL RL env
                     #   (multi-turn dialect runtime, reward shaping,
                     #   TRL adapter, golden / eval_suite / wikisql /
                     #   bird / synsql data sources). See
                     #   train/env/README.md and train/winning_runs.txt.
tests/               # pytest suite (oracles, harness, golden queries,
                     #   codegen pipeline, registry, property fuzzing,
                     #   cross-dialect differential, ...)
```

## Headline results

### Engine size

The full shared executor — every dialect on disk, every benchmark
question, every RL rollout flows through this code — is **1,500 lines**.
Adding the closed Tier-A IR it dispatches over puts the entire shared
runtime at **2,468 lines**:

| File | Lines |
|---|---:|
| [`manysql/executor/engine.py`](manysql/executor/engine.py) | 810 |
| [`manysql/executor/expr_eval.py`](manysql/executor/expr_eval.py) | 679 |
| [`manysql/executor/__init__.py`](manysql/executor/__init__.py) | 11 |
| [`manysql/ir/`](manysql/ir/) (plan, expr, types, printer) | 968 |
| **Total — shared runtime** | **2,468** |

Adding a *dialect* adds a `grammar.lark` and a `lowering.py`; the
executor doesn't change. That's the single-engine architecture
collapsing what would otherwise be N forks.

### Engine performance — parity with real Postgres at 200k rows

[`eval/perf_bench.py`](eval/perf_bench.py) runs a 15-query, dialect-
neutral SQL suite against (a) embedded Postgres 16 via `psycopg` and
(b) manysql `postgres_clone` (Lark parse → IR lowering → Polars
execution) on identical data with identical warmup/repeat schedules.

| dataset | repeats | geomean (manysql / pg) | row-equivalence |
|---|---|---|---|
| 50k rows | 1 warmup + 5 timed | 3.76× (Postgres faster) | 15 / 15 |
| **200k rows** | **2 warmup + 10 timed** | **1.03× (parity)** | **15 / 15** |

The crossover: manysql carries a fixed ~4–8 ms per-query overhead
(parse + lower + Polars plan build) that dominates on small data;
Polars's vectorized execution scales better than Postgres's row-based
engine, so by 200k rows the two are essentially tied — with each
winning on different query shapes (manysql ~7–9× on high-cardinality
`COUNT(DISTINCT)` and `GROUP BY`; Postgres ~3–4× on planner-friendly
semi-joins and constant-substring aggregations).

Full per-query breakdown: [`eval/PERFORMANCE.md`](eval/PERFORMANCE.md).

This number is the floor we needed: codegen-validation batteries
finish in seconds (gate every codegen attempt), RL rollouts can do
thousands of executions per minute (re-execute every model SQL through
the dialect runtime instead of relying on string-match), and
`manysql-eval --backend synthetic` is a viable closed-source-engine
stand-in.

### Training run results — synsql DDP-3, Qwen3-4B-Instruct GRPO

Two 500-step GRPO runs (recipe + commands in
[`train/winning_runs.txt`](train/winning_runs.txt)) share everything
except the dialect mix, on identical hyperparams (cosine LR 5e-6,
`num_generations=6`, `per_device_train_batch_size=2`, temperature 1.6,
`max_seq_length=6144`, LoRA rank 32, 3-rank DDP on 3× H100, vLLM
colocated):

- **1-d run**: synsql 1k slice (simple+moderate), one dialect =
  `snowflake_clone`. 51:47 wall-clock.
- **10-d run**: same slice, mix = `snowflake_clone` plus 9 generated
  gen-1 dialects (`sqlite_clone, bigmaria_pivot,
  bqserver_qualify_filter, mariflake_comma_semi, mydb2_rollup,
  mysqlite_upsert, orashift_merge, pgserver_schema_ns, redgres_lateral`).
  49:50 wall-clock.

Both adapters were evaluated against `eval/serve_lora.py` on the
50-question github_events benchmark with `--prompt-mode tag`, on
`snowflake_clone` (in-distribution for both runs) and `snowacle_qualify`
(a held-out gen-1 dialect neither LoRA saw):

| Model | Dialect | matched/50 |
|---|---|---:|
| Qwen3-4B base | `snowflake_clone` (in-dist) | 26 (52%) |
| 1-d LoRA | `snowflake_clone` (in-dist) | 23 (46%) |
| **10-d LoRA** | `snowflake_clone` (in-dist) | **25 (50%)** |
| Qwen3-4B base | `snowacle_qualify` (gen-1 OOD) | 12 (24%) |
| 1-d LoRA | `snowacle_qualify` (gen-1 OOD) | **18 (36%)** |
| 10-d LoRA | `snowacle_qualify` (gen-1 OOD) | **18 (36%)** |

Three takeaways:

1. **Multi-dialect training is approximately free on the trained
   distribution.** The 10-d LoRA scores 25/50 on `snowflake_clone` vs
   the 1-d LoRA's 23/50 — slightly better, despite spending 90% of its
   training compute on the other 9 dialects. The "specialist" run did
   not specialise harder, and the "generalist" did not regress on the
   specialty.
2. **Breadth of training does not multiply OOD transfer.** Both LoRAs
   hit the same 18/50 = 36% on the held-out `snowacle_qualify`,
   +12 pp over base. Whatever transfers is "knowing how to write
   parseable SQL with retries on `github_events` in some manysql
   dialect" — not "knowing the snowflake-family dialect family in
   general."
3. **Both LoRAs regress slightly vs base on the in-distribution
   benchmark** (52% → 46–50%). The typical GRPO narrowing artefact:
   first-attempt rate climbs (to 58–66%), but the long tail of
   "questions the base happened to solve once" shrinks.

Strategy implication: **train one 10-d LoRA per generation batch, not
10 single-dialect LoRAs.** Same training cost, comparable per-dialect
accuracy, single artefact to ship.

Bug-found-during-evaluation writeup, base-model floor diagnostics, and
the empty-set artifact pitfall: [`eval/FINDINGS.md`](eval/FINDINGS.md).

## Bundled dialect specs

Curated `DialectSpec`s ship in the package (`manysql-codegen --list`):
6 hand-curated specs (`mild_postgres_ish`, `moderate_keyword_swap`,
`aggressive_alien`, `snowflake_clone`, `sqlite_clone`,
`postgres_clone` — the latter three are faithful real-engine clones,
`postgres_clone` being the engine-perf baseline).

The campaign generator (`manysql-codegen batch`) emits LLM-designed
hybrid dialects on top of these. A representative sample (each
`examples.sql` is the parse battery rendered as inspectable dialect
SQL — fastest way to feel a dialect's surface):

| Dialect | Inspired by | Headline divergences |
|---|---|---|
| [`firebird_click_wildcard`](manysql/dialects/firebird_click_wildcard/examples.sql) | firebird, clickhouse, sql_server | `*` / `?` LIKE wildcard chars (replacing `%` / `_`), `CONCAT(...)`-only string joining, bracket identifier quoting, `LEN`/`IFNULL`/`MID` aliases. |
| [`oracle_maria_modconcat`](manysql/dialects/oracle_maria_modconcat/examples.sql) | oracle, mariadb, db2 | `MOD` as infix keyword for modulo, `+` for string concat, `<=>` null-safe equality, `^=` not-equal, `EXCEPT`/`INTERSECT` bind tighter than `UNION`. |
| [`snowdb2_dotmod_nullsafe`](manysql/dialects/snowdb2_dotmod_nullsafe/examples.sql) | snowflake, db2, sql_server | `.` column wildcard (replaces `*`), `MOD(a, b)` function-style modulo, bracket identifiers, `OFFSET`/`FETCH` limits. |
| [`hive_teradata_nullsafe_wild`](manysql/dialects/hive_teradata_nullsafe_wild/examples.sql) | hive, teradata, mysql | `<=>` null-safe equality, `?` LIKE wildcard for single-char match, `\|\|` concat with typed coercion, `MOD` infix, backtick identifiers. |
| [`db2_oracle_fetchfirst`](manysql/dialects/db2_oracle_fetchfirst/examples.sql) | db2, oracle, sql_2008 | `FETCH FIRST N ROWS ONLY` row-limiting, `MINUS` keyword for `EXCEPT`, `UNIQUE` keyword for `DISTINCT`, sole `<>` not-equal, upper-case identifier fold. |

Plus more under `manysql/dialects/`. Campaign manifests at
`manysql/dialects/_campaigns/<id>.json`.

## RL training entry point

[`train/grpo_sql.py`](train/grpo_sql.py) is the GRPO fine-tuning entry
point (Unsloth + TRL + vLLM); [`train/env/`](train/env/) is the
multi-turn RL environment. Five task generators: `golden`
(cross-dialect translation), `eval_suite` (NL on `github_events`),
`wikisql`, `bird` (5 GB SQLite pack, selective extraction), `synsql`
(2.5 M questions, streamed incrementally — a 1k sample transfers
~5–10 MB instead of the full 9.36 GB). Multi-dialect curricula are
first-class via `--dialects a,b,c`. See
[`train/env/README.md`](train/env/README.md) for the env API and
[`train/winning_runs.txt`](train/winning_runs.txt) for the verbatim
reproduction commands of the headline experiment.
