# IR scope

**manysql generates SQL dialects only.** The IR is *relational algebra over batch, single-source, read-only data*. Generating non-SQL query languages (graph, path, pipeline, streaming, procedural) is explicitly out of scope and would require a different system with a different IR.

This is a deliberate prior over the dialect-generation space; this document records what is and is not representable so future feature additions go through the IR-extension RFC process rather than silent scope creep.

## Tier A — natively in v1 IR

The v1 operator set:

- `Scan` — read a base table from the catalog.
- `Project` — compute a list of named expressions.
- `Filter` — keep rows matching a predicate (also used to model `HAVING` and `QUALIFY`).
- `Join` — `INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`, plus `SEMI` / `ANTI` for semi-joins.
- `Aggregate` — `GROUP BY` plus aggregate functions.
- `Window` — add window-function columns to a relation.
- `Sort` — order rows.
- `Limit` — bound row count, with `OFFSET`.
- `Distinct` — set-of-rows projection.
- `SetOp` — `UNION` / `INTERSECT` / `EXCEPT`, with `ALL` flag.
- `WithCTE` — non-recursive CTEs.
- `RecursiveCTE` — single recursive CTE binding.
- `Apply` — dependent join, used to lower correlated subqueries.

Scalar expressions, aggregate calls, window calls, scalar/EXISTS/IN subqueries are all expression nodes (`manysql/ir/expr.py`).

All Tier-1 runtime semantic divergence (null ordering, division-by-zero, integer division, identifier folding, set-op default, NULL-safe equality, COUNT-on-empty, boolean truthiness, default window frame, string-concat operator, function aliasing, keyword aliasing, implicit-coercion matrix, LIKE case sensitivity) is expressed via `SemanticConfig` (see `manysql/spec/`) — **not** via IR shape.

This covers the bulk of practical Postgres ↔ Snowflake ↔ Databricks ↔ Trino divergence at the surface and silent-semantic layers.

## Tier B — additive IR extensions, RFC-gated

Planned for v1.5+, each via an explicit RFC under `manysql/ir/rfcs/`:

- **Typed-value extension** — arrays, structs, maps as values; new `Unnest` / `Flatten` nodes.
- **JSON** — path expressions; storage as a typed column.
- **Regex** — flavor selection (POSIX / PCRE / Java) as a SemanticConfig knob, with a `RegexMatch` expression node.
- **Sampling** — new `Sample` node (Bernoulli / system-block).
- **MERGE/UPSERT** — DML extension; only relevant if DDL/DML enters scope.
- **Pivot/Unpivot** — either sugar over `Aggregate` + `CASE`, or first-class nodes if dialects differ.
- **Time travel** — versioned `Scan` (`AT VERSION`, `BEFORE TIMESTAMP`).

These **grow** the IR by adding nodes. They do not redesign existing nodes.

## Tier C — outside the IR's prior

Cannot be represented in this IR without becoming a different IR. Generating dialects of these is **explicitly out of scope** for manysql; they would require a sibling system with a weaker structural assumption (e.g., a tagged AST + generic interpreter, no fixed algebra):

- Graph traversal languages (Cypher, GQL).
- Tree/path query languages (XPath, XQuery, jq).
- Pipeline-style data languages (KQL, PRQL chains).
- Event-time streaming queries (windowing over unbounded streams).
- Probabilistic queries.
- Procedural extensions with variables, loops, and exceptions (PL/SQL, T-SQL stored procedures).

## System-level divergence — case-by-case

Some "dialect features" are not query-language features at all but properties of the surrounding system:

- **Lazy vs eager evaluation** (Spark) — affects what queries *mean* in subtle ways.
- **Federation across catalogs** (Trino) — needs a connector model.
- **Storage-attached time travel** (Snowflake, Delta) — needs versioned storage.
- **Slot/cost-based query shape constraints** (BigQuery wildcard tables) — language-design pressure, not language semantics.

The IR can absorb any one via a Tier-B extension. Each is a different category and the IR will never elegantly capture all of them; we add them when a target dialect requires them.

## Process

Adding a node, expression, or type to the IR requires a short RFC under `manysql/ir/rfcs/` covering: the surface dialect features it enables, the executor changes required, the verification-oracle plan (which oracle(s) can verify the new feature), and any SemanticConfig knobs introduced.

The first such RFC is `manysql/ir/rfcs/0001-tier3.md` (arrays/structs/maps, JSON, regex flavor + deep date/time), gated for v1.5.

This gate exists because every IR addition is a contract honored by the executor, every oracle, and every generated dialect's lowering. Silent additions break that invariant.
