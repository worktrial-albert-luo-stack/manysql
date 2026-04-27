# manysql

A generator of synthetic **SQL dialects** with full per-dialect query
engines, plus the verification and benchmarking infrastructure to use them
as LLM training data.

manysql is **scope-locked to SQL**. Generating non-SQL query languages
(Cypher, jq, KQL, PRQL, streaming, procedural) is explicitly out of scope —
see `manysql/ir/SCOPE.md`.

## What's in the box

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
              └─►  manysql-eval benchmark
                   (NL question → LLM SQL → executor → score against reference,
                   pluggable across SQLite / Tinybird / synthetic-dialect backends)
```

Given a `DialectSpec` describing how a target dialect diverges from a
near-ANSI reference (keyword aliases, surface knobs like `LIMIT` vs
`OFFSET ... FETCH`, function aliases, semantic knobs like null ordering
or division-by-zero), `manysql-codegen` emits a runnable engine for that
dialect:

- `grammar.lark` — Lark grammar for the dialect's surface syntax.
- `lowering.py` — parse tree → shared logical-plan IR.
- `semantics.json` — runtime knobs the executor honors.
- `overrides.py` — optional Python implementations for novel functions/
  operators no canonical executor handler can express.
- `passes.py` — optional Plan→Plan rewrites between lowering and
  execution (for surfaces that need non-canonical IR markers desugared).
- `effects.py` — optional named handlers swapped into executor decision
  points (for runtime divergences whose space isn't a small closed enum).
- `metadata.json` — provenance, retry log, and lifecycle state.
- `battery.json` + `examples.sql` — the parse/IR battery in dialect surface,
  rendered as inspectable SQL.

The pipeline starts with deterministic emitters (templated from the spec)
and falls back to an LLM refine loop when the spec's surface needs
structural changes the templates can't express. LLM output is rolled back
automatically if it regresses either battery.

## Architecture

### IR (Tier-A, scope-locked)

`manysql/ir/` is relational algebra over batch, single-source, read-only
data. v1 operators: `Scan`, `Project`, `Filter`, `Join` (incl. SEMI/ANTI),
`Aggregate`, `Window`, `Sort`, `Limit`, `Distinct`, `SetOp`, `WithCTE`,
`RecursiveCTE`, `Apply` (dependent join for correlated subqueries).
Scalar / aggregate / window calls plus scalar/EXISTS/IN subqueries are
expression nodes.

All Tier-1 *runtime* divergence (null ordering, division-by-zero, integer
division, identifier folding, set-op default, NULL-safe equality,
COUNT-on-empty, boolean truthiness, default window frame, string-concat
operator, function/keyword aliasing, LIKE case sensitivity, GROUP BY/SELECT
scope rules) lives in `SemanticConfig` — **not** in IR shape. Two dialects
with the same Plan but different `SemanticConfig` legitimately produce
different rows.

Tier-B IR extensions (arrays/structs, JSON, regex flavor, sampling, time
travel, MERGE, pivot/unpivot) are deferred to v1.5 behind RFCs under
`manysql/ir/rfcs/`. Tier-C languages (Cypher, jq, streaming, procedural)
are out of scope by design.

See `manysql/ir/SCOPE.md` for the full prior.

### Executor

`manysql/executor/` is a single Polars/PyArrow executor that takes
`(Plan, SemanticConfig, catalog)` and returns a `pl.DataFrame`. Every
behavior decision point reads from `SemanticConfig`, so swapping configs
swaps semantics. Optional per-dialect overrides, passes, and effects
extend it without forking the executor.

### Verification harness

`manysql/oracle/` runs a Plan through every applicable oracle and compares:

- `DuckDBOracle` and `SQLiteOracle` — render the Plan back to SQL and
  execute it on a real engine.
- `ReferenceInterpreter` — slow, hand-coded Python interpreter over lists
  of dicts. Independent code path from the Polars executor; agreement
  between the two is the strongest local signal of correctness.
- `PropertyOracle` — structural invariants (Distinct deduplicates, Sort
  is sorted, Limit limits, Aggregate-without-GROUP-BY produces exactly
  one row, SEMI/ANTI doesn't leak right-side columns, etc.). Always
  applicable; cheap; complements row-producing oracles.
- `CrossDialectOracle` — runs the *same logical query* through two or
  more generated dialects and flags disagreement as either a curated
  semantic-divergence training example or a codegen bug.

The harness picks a primary by confidence, runs all applicable oracles,
and returns one of `PASS / FAIL / NEEDS_REVIEW / NO_ORACLE`. Inter-oracle
disagreement is `NEEDS_REVIEW` (informative — usually means the plan hits
a corner case where engines themselves diverge).

`tests/test_property_hypothesis.py` Hypothesis-fuzzes random catalogs
against the property oracle for every golden plan.

### Dialect registry

`manysql/dialects/registry.py` is a backend-swappable store with a
first-class lifecycle (`DRAFT → GENERATING → GENERATED → VALIDATED /
NEEDS_REVIEW / FAILED → DEPRECATED`). v1 backend is on-disk Python
packages under `manysql/dialects/<name>/`. Validation runs append to a
per-dialect history.

### Storage and golden queries

`manysql/storage/` ships a deterministic Parquet catalog (employees /
departments / regions / sales / categories tree) designed to exercise
nullables, ties, leap days, correlated subqueries, recursive CTEs, and
empty groups. `manysql/golden/queries.py` is the hand-curated SQL
corpus the parse/IR batteries and harness exercise.

## Tooling

Three CLI entry points (all installed by `uv sync`):

| Command | Purpose |
| --- | --- |
| `manysql-codegen gen <spec>` | Materialize a dialect package from a `DialectSpec` (or one of the bundled examples). `--use-llm` runs at least one LLM refinement pass on top of the deterministic baseline. |
| `manysql-codegen batch --n N --prior <vibe>` | Outer-loop campaign: design N diverse specs (sequentially, with a running ledger for diversity) from a free-form prior + structured knobs, then fan them through the codegen pipeline in parallel. Manifest at `manysql/dialects/_campaigns/<id>.json`. Requires an LLM key. |
| `manysql-dialect diff <name>` | Side-by-side diff of a generated dialect's reskinned battery vs. the canonical reference SQL. Useful for "did this surface knob actually take effect?" |
| `manysql-eval` (alias `eval`) | Pluggable LLM SQL benchmark: NL question → model SQL → execute → retry on error → score. Backends: `sqlite` (default, with a synthetic GitHub-events seed), `tinybird`, `synthetic` (a manysql-generated dialect with a SQLite reference auto-attached for ground truth). LLM providers: `openai`, `openrouter`, `vllm`. See `eval/README.md` for the full surface. |

### Bundled dialect specs

These are the curated `DialectSpec`s that ship in the package, exposed
via `manysql-codegen gen <name>`:

```bash
manysql-codegen --list
```

| Name | Divergence | Notes |
| --- | --- | --- |
| `mild_postgres_ish` | mild | Surface stays ANSI; flips a handful of semantic knobs (lowercase fold, NULLS FIRST default DESC, integer division truncates, division-by-zero errors). Smoke test for the codegen pipeline. |
| `moderate_keyword_swap` | moderate | Renamed clause keywords + alternate `LIMIT` syntax. |
| `aggressive_alien` | aggressive | NIL nulls, `::` casts, `~=` for null-safe equality, `+` for string concat, `OFFSET … FETCH` limits, no ILIKE, `HAVE` instead of `HAVING`, `ORDERED BY` instead of `ORDER BY`. Stresses the LLM refine loop. |
| `snowflake_clone` | mild | Faithful Snowflake target: UPPER identifier fold, ILIKE, NULLS FIRST/LAST defaults, division-by-zero errors, integer division promotes to float, `//` line comments, `TRY_CAST`/`NVL`/`IFNULL` aliases. |
| `sqlite_clone` | mild | Faithful SQLite target: preserve-case (ASCII-insensitive) identifiers, NULL-on-divide-by-zero, truncating integer division, case-insensitive `LIKE`, no `ILIKE`, C-style boolean truthiness, `IFNULL`/`SUBSTRING` aliases. Pairs with `snowflake_clone` to bracket realistic real-world divergence. |

### Synth-generated dialects (sample)

The batch campaign generator (`manysql-codegen batch`) emits LLM-designed
dialects on top of the bundled specs above; running a few campaigns has
populated `manysql/dialects/` with ~30 distinct surfaces. A non-exhaustive
sample of the more recognizable ones, to give a sense of the design
space the codegen pipeline can hit:

| Name | Inspired by | Headline divergences |
| --- | --- | --- |
| `mild_postgres_core` | postgres | `::` casts, double-quoted identifiers, lowercase fold, `ILIKE`, `\|\|` concat, `EXTRACT`/`DATE_PART` temporals. |
| `mild_snowflake_upper` | snowflake | Upper-case identifier folding, `ILIKE`, NULLS-last default ordering. |
| `moderate_mysql_loose` | mysql | Backtick identifiers, `LIMIT` w/o `OFFSET` keyword, loose boolean truthiness, `CONCAT(...)`-only string joining. |
| `tsql_ish` | sql_server | Bracket identifiers, `+` string concat, `LEN`, case-insensitive default collation (effects.py lane). |
| `bracket_oracle_pg` | sql_server, oracle, postgres | Bracket-quoting + Oracle `FETCH FIRST` row limiting + Postgres `::` casts, `CONCAT`-only string joining. |
| `snowacle_qualify` | snowflake, oracle | `QUALIFY`-only row limiting, NULLS FIRST as immutable global default, `DEFINE` instead of `WITH`, `MATCHING` instead of `USING`. |
| `bigmaria_pivot` | bigquery, mariadb | `PIVOT`/`UNPIVOT` as first-class FROM-clause operators, aggregate `FILTER` spelled as `WHERE` inside parens. |
| `redgres_lateral` | redshift, postgres, bigquery | Mandatory `LATERAL` for correlated FROM subqueries, `MATCHING` instead of `USING`, `CONCAT_WS`-only string joining. |
| `mariflake_comma_semi` | mariadb, snowflake, db2 | Semicolons between comma-joined tables, `DEFINE` for CTEs, reversed `EXCEPT`/`INTERSECT` precedence over `UNION`. |
| `mysqlite_upsert` | mysql, sqlite, snowflake | `UPSERT` keyword for MERGE, reversed `EXCEPT`/`INTERSECT` precedence, `MATCHING` for join `USING`. |
| `pgserver_schema_ns` | postgres, sql_server | Schema-qualified function namespaces (`schema::func`), `OFFSET`/`FETCH` limits, reversed `<>` not-equal. |
| `snowflake_server_oracle_neq` | snowflake, sql_server, oracle | Exclusive `^=` not-equal operator, `CONVERT_FN(...)` casting, double-quoted strings, upper-fold identifiers. |
| `db2_snowflake_sqlite_rangefold` | db2, snowflake, sqlite | `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` window default, double-quoted identifiers with upper fold. |

Campaigns themselves get manifests under
`manysql/dialects/_campaigns/<id>.json` (per-campaign config, brief, every
drafted spec, and per-package status); the registry skips that directory
because it contains no dialect packages. Run
`manysql-codegen batch --n 5 --prior "..."` to generate your own.

## Repository layout

```
manysql/
├── ir/              # logical-plan IR + SCOPE.md + RFCs/
├── executor/        # Polars/PyArrow IR executor
├── oracle/          # DuckDB / SQLite / reference interpreter / property /
│                    #   cross-dialect oracles + harness
├── storage/         # Parquet-backed deterministic test catalog
├── spec/            # SemanticConfig + DialectSpec schemas (Pydantic)
│   └── examples/    # bundled DialectSpec instances
├── dialects/
│   ├── _reference/  # hand-written near-ANSI / DuckDB-aligned reference
│   ├── _campaigns/  # batch-codegen manifests (config + drafted specs +
│   │                #   per-package status; one JSON per `batch` run)
│   ├── <name>/      # generated dialect packages (one folder each)
│   ├── registry.py  # lifecycle-aware, backend-swappable registry
│   ├── card.py      # shared dialect-card prompt (used by eval + train)
│   └── diff.py      # battery diff helpers
├── codegen/         # deterministic emitters + grammar/lowering LLM agents
│                    #   + parse battery + IR-equivalence battery
├── golden/          # hand-curated SQL corpus (queries.py)
├── llm/             # thin OpenAI/OpenRouter/Anthropic chat client
└── verify/          # harness loops over goldens

eval/                # pluggable LLM SQL benchmark (see eval/README.md)
tests/               # pytest suite (oracles, harness, golden queries,
                     #   codegen pipeline, registry, property fuzzing,
                     #   cross-dialect differential, ...)
train/               # GRPO/RLVR training entrypoints + SQL RL env
                     #   (multi-turn dialect runtime, reward shaping,
                     #   TRL adapter, WikiSQL / BIRD-SQL / SynSQL-2.5M
                     #   data sources). See train/env/README.md for the
                     #   full surface.
```

## RL training over synthetic dialects

`train/env/` is a multi-turn RL environment over manysql synthetic
dialects: agent reads a system prompt + a question, calls a `run_sql`
tool, and either matches the gold rows (success) or gets a parse /
runtime error trace it can iterate on next turn. Reward is a function
of correctness *and* the number of turns it took to get there. See
`train/env/README.md` for the architecture and adapter API.

`train/grpo_sql.py` is the GRPO fine-tuning entry point (Unsloth + TRL
+ vLLM). Five task generators ship:

| Generator | Source | Best for |
|---|---|---|
| `golden` (default) | Cross-dialect translation tasks built from `manysql.golden.queries` on the canonical 5-table catalog. | Teaching the model to *speak* a synthetic dialect (mechanical translation, no NL ambiguity). |
| `eval_suite` | NL questions from `eval.dataset.questions` over the synthetic `github_events` corpus. | End-to-end NL→SQL on the same benchmark as `manysql-eval`. |
| `wikisql` | NL/table/answer triples sampled from `Salesforce/wikisql`. Each task carries its own small Wikipedia table; column names are sanitized to safe `c_*` identifiers. | NL→SQL on real, diverse schemas (1k–80k tasks). |
| `bird` | NL/multi-table-database/answer triples from BIRD-SQL (`birdsql/bird23-train-filtered` for train, `bird_sql_dev_20251106` for dev). Each task references one real Kaggle-style database (5–25 tables) plus a domain-`evidence` field. Strictly harder than WikiSQL (joins, subqueries, CTEs, windows, evidence-driven reasoning). | NL→SQL once the model ceilings out on WikiSQL — the simple+moderate slice keeps yielding signal for 4B+ instruct models. |
| `synsql` | NL/multi-table-database/answer triples from `seeklhy/SynSQL-2.5M` (2.54M questions across 16k+ synthetic SQLite DBs, with `simple`/`moderate`/`complex`/`highly complex` complexity bands and a wide spread of NL question styles). Streams just the requested subset of the 9.36 GB `data.json` over HTTPS instead of downloading the full file. | NL→SQL at million-question scale — strictly larger and more diverse than BIRD; useful when GRPO ceilings out on BIRD's simple+moderate slice. |

Multi-dialect curricula are first-class: pass
`--dialects aggressive_alien,mild_postgres_ish,tsql_ish` and the same
training run trains one model on all three at once. Each row's system
prompt tags it with `Dialect: <name>` so the model learns to route;
the reward function dispatches per row using the dataset's `dialect`
column and re-executes against ground truth. Two coverage modes:

| Mode | Effect |
|---|---|
| `partition` (default) | Round-robin assign each task to one dialect (N rows total). |
| `cross_product` | Emit each task once per dialect (N × M rows; `task_id` suffixed `__<dialect>`). |

Each dialect's system prompt includes its **dialect card**
(`manysql.dialects.card.render_dialect_card`): surface divergences,
canonical patterns, function aliases, and a few worked examples — same
priors `manysql-eval` uses, by design.

```bash
# CPU dry-run (no GPU, no Unsloth/vLLM): exercises dataset + tool +
# every reward-component end-to-end on a handful of synthetic
# completions. Prints the reward breakdown + multi-dialect dispatch.
uv run python -m train.grpo_sql \
    --dialects aggressive_alien,tsql_ish \
    --generator wikisql --wikisql-size 32 \
    --dry-run

# GPU box: real training. See train/grpo_sql.py module docstring for
# the full uv install incantation (unsloth + trl + vllm + transformers).
python train/grpo_sql.py \
    --dialects aggressive_alien,mild_postgres_ish,tsql_ish \
    --generator wikisql --wikisql-size 2000 \
    --max-steps 500

# BIRD: harder than WikiSQL. Auto-downloads a ~5GB SQLite pack on
# first run (cached at ~/.cache/manysql/bird/train/, with selective
# extraction so only the DBs referenced by your sample land on disk).
# Filter to simple+moderate difficulty by default; 'challenging' is
# opt-in via --bird-difficulty. Memory- and disk-safety knobs:
#   --bird-max-db-bytes BYTES   skip oversized DBs (default 500MB,
#                               excludes the donor=4.5GB long tail)
#   --bird-max-rows-per-table N cap each table to N rows via a
#                               sampled SQLite mirror (default 200k)
python train/grpo_sql.py \
    --dialects aggressive_alien,mild_postgres_ish,tsql_ish \
    --generator bird --bird-size 1000 \
    --max-steps 500

# SynSQL-2.5M: million-question synthetic corpus. The 9.36GB data.json
# is streamed (custom incremental JSON parser) so a 1k-sample run
# transfers ~5-10MB instead of the full file; databases.zip (55MB) is
# downloaded once and selectively extracted. Cached subsets live under
# ~/.cache/manysql/synsql/. Same complexity / row-cap knobs as BIRD:
#   --synsql-complexity simple,moderate    (default; or 'complex' / 'highly complex')
#   --synsql-split train|dev|test          (carved by absolute item index)
#   --synsql-max-rows-per-table N          (default 200k; rarely triggers — DBs are tiny)
python train/grpo_sql.py \
    --dialects aggressive_alien,mild_postgres_ish,tsql_ish \
    --generator synsql --synsql-size 2000 \
    --max-steps 500
```

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

# 3. Sanity-check the eval backend can load it.
uv run manysql-eval --backend synthetic \
    --synthetic-dialect mild_postgres_ish \
    --limit 5 --dry-run

# 4. Run a real eval. Reference SQL is computed via the auto-attached
#    SQLite executor; the candidate dialect engine parses + executes
#    the LLM's reply.
uv run manysql-eval --backend synthetic \
    --synthetic-dialect mild_postgres_ish \
    --provider openai --model gpt-4o-mini \
    --limit 10 -j 4
```

For an aggressive surface where the deterministic emitter alone won't
suffice, force the LLM refine loop:

```bash
uv run manysql-codegen gen aggressive_alien --use-llm --overwrite
```

To populate a benchmark with many synthetic dialects in one shot:

```bash
uv run manysql-codegen batch --n 5 \
    --prior "variants between mssql and snowflake" \
    --theme mixed \
    --inspired-by sql_server,snowflake
```

The pytest suite covers the IR, executor, every oracle, the harness, the
registry lifecycle, the codegen pipeline (incl. agent stubs), property
fuzzing, and cross-dialect differential checks:

```bash
uv run pytest -n auto
```

## v1 scope

ANSI-core read-only + window functions + CTEs (incl. recursive) + set ops
+ subqueries (incl. correlated). Arrays / structs / JSON / regex flavor /
sampling / time travel / MERGE / pivot are deferred to v1.5 via the
IR-extension RFC process under `manysql/ir/rfcs/`.

The first such RFC is `manysql/ir/rfcs/0001-tier3.md` (arrays/structs/maps,
JSON path expressions, regex flavor selection, deep date/time).
