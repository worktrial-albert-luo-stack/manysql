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
| `manysql-codegen <spec>` | Materialize a dialect package from a `DialectSpec` (or one of the bundled examples). `--use-llm` runs at least one LLM refinement pass on top of the deterministic baseline. |
| `manysql-dialect diff <name>` | Side-by-side diff of a generated dialect's reskinned battery vs. the canonical reference SQL. Useful for "did this surface knob actually take effect?" |
| `manysql-eval` (alias `eval`) | Pluggable LLM SQL benchmark: NL question → model SQL → execute → retry on error → score. Backends: `sqlite` (default, with a synthetic GitHub-events seed), `tinybird`, `synthetic` (a manysql-generated dialect with a SQLite reference auto-attached for ground truth). LLM providers: `openai`, `openrouter`, `vllm`. See `eval/README.md` for the full surface. |

### Bundled dialect specs

```bash
manysql-codegen --list
```

| Name | Divergence | Notes |
| --- | --- | --- |
| `mild_postgres_ish` | mild | Surface stays ANSI; flips a handful of semantic knobs (lowercase fold, NULLS FIRST default DESC, integer division truncates, division-by-zero errors). Smoke test for the codegen pipeline. |
| `moderate_keyword_swap` | moderate | Renamed clause keywords + alternate `LIMIT` syntax. |
| `aggressive_alien` | aggressive | NIL nulls, `::` casts, `~=` for null-safe equality, `+` for string concat, `OFFSET … FETCH` limits, no ILIKE, `HAVE` instead of `HAVING`, `ORDERED BY` instead of `ORDER BY`. Stresses the LLM refine loop. |

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
│   ├── <name>/      # generated dialect packages (one folder each)
│   ├── registry.py  # lifecycle-aware, backend-swappable registry
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
train/               # GRPO/RLVR training entrypoints (template adapted
                     #   from a GSM8K reference; SQL adaptation hooks
                     #   marked inline)
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env  # add OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY as needed
```

## End-to-end smoke test

```bash
# 1. Generate a dialect package (deterministic, ~instant).
uv run manysql-codegen mild_postgres_ish

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
uv run manysql-codegen aggressive_alien --use-llm --overwrite
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
