# manysql

A generator of synthetic **SQL dialects** with full per-dialect query engines, intended to produce LLM training data that teaches generalization across SQL surface and semantics.

manysql is **scope-locked to SQL**. Generating non-SQL query languages (Cypher, jq, KQL, PRQL, streaming, procedural) is explicitly out of scope — see `manysql/ir/SCOPE.md`.

## What it does

Given a `DialectSpec` describing a SQL dialect (keyword aliases, function library, semantic-knob overrides, novel features), `manysql` generates a complete runnable engine for that dialect:

- **Grammar** (Lark) — surface syntax of the dialect.
- **Lowering** — AST → shared logical-plan IR.
- **Semantic config** — runtime knobs (null ordering, division-by-zero, identifier folding, etc.) honored by the executor.
- **Optional operator overrides** — Python implementations for novel semantics no shared operator can express.

The shared infrastructure (logical-plan IR, Polars/PyArrow executor, multi-oracle verification harness) is hand-written. Per-dialect frontends are LLM-generated and validated by the harness.

## Architecture at a glance

- `manysql/ir/` — logical-plan IR (relational algebra over batch read-only data).
- `manysql/executor/` — Polars/PyArrow IR executor, parameterized by `SemanticConfig`.
- `manysql/oracle/` — multi-oracle verification: DuckDB, SQLite, hand-written Python reference IR interpreter, cross-dialect differential, property-based.
- `manysql/storage/` — Parquet-backed test datasets.
- `manysql/spec/` — `SemanticConfig` and `DialectSpec` schemas (Pydantic).
- `manysql/dialects/_reference/` — hand-written near-ANSI reference dialect.
- `manysql/dialects/<name>/` — generated dialect packages.
- `manysql/dialects/registry.py` — backend-swappable dialect registry with first-class lifecycle states.
- `manysql/codegen/` — LLM-driven generation pipeline.
- `manysql/verify/` — golden IR plans + harness loops.
- `eval/` — pluggable LLM SQL benchmark (OpenAI / OpenRouter / vLLM × SQLite / Tinybird / synthetic). See `eval/README.md`.
- `tests/` — pytest suite.

See `manysql/ir/SCOPE.md` for what the IR can and cannot represent.

## Setup

```bash
uv sync --extra dev
cp .env.example .env  # add OPENAI_API_KEY and/or OPENROUTER_API_KEY
```

## v1 scope

ANSI-core read-only + windows + CTEs (incl. recursive) + set ops + subqueries (incl. correlated). Arrays/structs/JSON/regex deferred to v1.5 via IR-extension RFC.
