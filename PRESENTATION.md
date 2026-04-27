# manysql

**A generator of synthetic SQL dialects + the verification harness and RL
environment to use them as LLM training data.**

> Audience: engineers familiar with SQL but not necessarily with query
> engine internals. Goal: by the end, you should be able to predict which
> design decision was forced by which property of SQL.

Outline:

1. SQL language properties (and why they matter for what comes next)
2. Environment design — options considered, and why we settled on
   generic engine + deterministic knobs + LLM lanes
3. *(later sections — codegen pipeline, oracle harness, RL env, eval, results)*

---

## 1. SQL language properties

The design decisions that show up in section 2+ are forced by *facts about SQL*,
not by taste. This section is the prior. Five claims:

1. SQL is layered: **lexical → syntactic → semantic**. Each layer varies
   independently across real-world dialects.
2. **Surface ≠ semantics.** Two queries that look character-for-character
   identical can return different rows on different engines.
3. The silent-semantic divergence catalog is real, recurring, and *small*.
4. **Relational algebra is the common substrate** — every Tier-A SQL
   dialect lowers to ~14 operators.
5. SQL has a **clean scope boundary**. Cypher / jq / streaming / procedural
   are not SQL.

---

### 1.1 SQL is layered

Three roughly-independent layers of variation:

| Layer | What varies | Examples |
|---|---|---|
| **Lexical** | tokens, comments, quotes, identifier folding | `--` vs `#` vs `//` comments; `"id"` vs `` `id` `` vs `[id]`; lowercase vs UPPERCASE folding of bare identifiers |
| **Syntactic** | clause keywords, operator spellings, clause shapes | `LIMIT n` vs `OFFSET … FETCH NEXT n` vs `TOP n`; `||` vs `+` vs `CONCAT(...)`; `CAST(x AS T)` vs `x::T` vs `CONVERT(T, x)` |
| **Semantic** | how the same plan executes against the same data | `NULL` ordering on `ORDER BY`; `1/0` returning `NULL` vs erroring; `COUNT(*)` on empty group returning 0 vs NULL |

> **Implication for the design:** these layers can be parameterized
> independently. A change to identifier folding doesn't ripple into
> grammar; a change to `LIMIT` syntax doesn't ripple into the executor.
> This is what lets us split codegen into a small set of orthogonal
> knobs in section 2.

---

### 1.2 Surface ≠ semantics

The same query, byte-for-byte, on the same data, can return different rows.

```sql
SELECT id, name FROM employees ORDER BY manager_id;
-- Postgres:   NULLs LAST  on ASC by default
-- MySQL:      NULLs FIRST on ASC by default

SELECT 1 / 0;
-- Postgres:   ERROR: division by zero
-- MySQL:      NULL
-- Snowflake:  ERROR
-- (no engine returns +Inf, but the IEEE-754 path exists for floats)

SELECT * FROM t WHERE name LIKE 'Alice';
-- Postgres:   case-sensitive
-- SQLite:     case-insensitive (ASCII)
-- MySQL:      depends on collation

SELECT 5 / 2;
-- Postgres:   2     (integer division, truncating)
-- MySQL:      2.5   (always promotes)
-- SQLite:     2     (truncating)
```

> **Implication:** a "translate to dialect X" approach that only rewrites
> tokens (a transpiler) cannot capture this. The model has to *know* the
> semantic divergence; the system has to *verify* it executed against
> the right semantics. This kills option (a) in section 2.

---

### 1.3 The silent-semantic divergence catalog is small and enumerable

Across Postgres / MySQL / SQLite / Snowflake / DuckDB / SQL Server / Oracle /
DB2 / Trino / Spark, the recurring axes of *runtime* divergence form a
~15-knob list:

| Axis | Closed enum |
|---|---|
| Identifier case folding | lower / upper / preserve |
| Null order on ASC / DESC | first / last (per direction) |
| Divide by zero | null / error / inf |
| Integer division | truncate / promote |
| `LIKE` case sensitivity | sensitive / insensitive |
| `ILIKE` supported | yes / no |
| String concat operator | `\|\|` / `+` / `CONCAT()`-only |
| Default for `UNION`/`INTERSECT`/`EXCEPT` | distinct / all |
| Boolean truthiness | strict / C-style |
| `COUNT(DISTINCT NULL)` | excluded / included |
| `SUM` on empty | null / 0 |
| Window default frame | range / rows / unbounded |
| `GROUP BY <select alias>` accepted | yes / no |
| Set-op precedence | ANSI / `INTERSECT`-tighter |
| Null-safe equality | supported / not |

This is the entire `SemanticConfig` (`manysql/spec/semantics.py`). 15 small
enums × ~3 values each = the **closed-world knob surface** that covers the
bulk of practical Postgres ↔ Snowflake ↔ Databricks ↔ Trino divergence.

> **Implication:** if the divergence space were continuous or
> domain-specific, we'd be stuck with per-dialect engines. Because it's
> enumerable, one parameterized executor handles all of them. The engine
> reads from `SemanticConfig` at every decision point.

---

### 1.4 But the long tail is open-world

Not everything is a small enum. The escape hatches we'll need:

- **Collations** — `Latin1_General_CI_AS` is one of dozens; you don't
  enum them, you ship a comparator.
- **Custom functions** — `TRY_CAST`, `NVL`, `IFNULL`, `MID`, `STRPOS`,
  `INSTR`. Names alias each other but bodies sometimes differ.
- **Plan-shape sugar** — `LIMIT N WITH TIES`, `QUALIFY`, `PIVOT`. The
  *intent* is expressible in canonical IR, but the surface produces a
  marker the executor doesn't recognize.
- **Locale-sensitive comparison, custom rounding modes, etc.**

> **Implication:** alongside the closed-world knobs, we need a small
> number of *open-world per-dialect lanes* — Python modules a dialect
> can drop in to extend the runtime. This becomes `lowering.py`,
> `overrides.py`, `passes.py`, `effects.py` in section 2.

---

### 1.5 Relational algebra is the common substrate

Across every Tier-A SQL dialect, the IR is the same ~14 operators:

```
Scan, Project, Filter, Join (inc. SEMI/ANTI),
Aggregate, Window, Sort, Limit, Distinct,
SetOp (UNION/INTERSECT/EXCEPT), WithCTE, RecursiveCTE, Apply
```

A `SELECT … FROM … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT` is just
sugar for `Limit(Sort(Filter(Aggregate(Filter(Scan)))))`. The shape of
that tree does not change between Postgres and Snowflake — only the
runtime config does.

> **Implication:** one IR + one executor + per-dialect parsers and
> lowerings is enough. No N-engines fork. Dialect divergence lives in
> data (`SemanticConfig`) plus four small per-dialect modules — not in
> the executor's source code.

---

### 1.6 SQL has a clean scope boundary

What the IR cannot represent (and we're explicit it never will):

- Graph traversal (Cypher, GQL)
- Tree / path queries (XPath, jq)
- Pipeline-style (KQL, PRQL)
- Streaming / event-time
- Procedural (PL/SQL, T-SQL stored procs with loops, exceptions)

These need a different IR with a weaker structural assumption. We
explicitly refuse to generate them — `manysql/ir/SCOPE.md` is the
contract.

> **Implication:** by scope-locking *up front*, every later design
> decision can assume "the input is a Tier-A SQL plan." That assumption
> would not hold if we tried to be a generic "query language generator,"
> and the rest of the system would be 5× larger and worse.

---

### 1.7 Section recap — what the rest of the design has to honor

| SQL property | Force on the design |
|---|---|
| Surface ≠ semantics | Token-level transpilation isn't enough. The system must *execute* candidate SQL under the dialect's actual runtime. |
| Layered (lex / syn / sem) | Codegen should parameterize the layers independently — orthogonal knobs. |
| Closed-world divergence catalog (~15 axes) | A small enum-driven config can cover most of it (`SemanticConfig`). |
| Long tail exists (collations, novel funcs, plan sugar) | Need per-dialect Python escape hatches alongside the enums. |
| Rel-algebra substrate | One IR, one executor, parameterized — not N forks. |
| Clean scope boundary | Lock IR to Tier-A; refuse out-of-scope dialects by design. |

Hold these in your head — section 2 will name each one as it shows up.

---

## 2. Environment design

The problem we're trying to solve, very concretely:

> Given a `DialectSpec` describing how a dialect diverges from ANSI,
> produce something we can:
>
> 1. **Parse** SQL written in that dialect,
> 2. **Execute** that SQL against in-memory data and get *correct* rows,
> 3. **Verify** that the execution agreed with what the dialect was
>    supposed to mean,
>
> at low marginal cost per dialect, scaling to hundreds of synthetic
> dialects, so an LLM can be trained / evaluated against them.

We considered four architectures before landing on the fifth. Each had a
fatal property given the SQL properties from section 1.

---

### 2.1 Option A — Transpilers (rewrite ANSI SQL into the target surface)

**Idea:** the LLM emits ANSI SQL; we lexically rewrite it to the synthetic
dialect's surface; we run the rewritten string on a real engine
(Postgres, DuckDB, SQLite).

Concretely: `LIMIT 10` → `OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY`,
`||` → `+`, `CAST(x AS INT)` → `x::INT`, etc.

**Why it fails (against properties from section 1):**

1. **Surface ≠ semantics (1.2).** A transpiler can rewrite tokens, but
   it can't fix the model's mental model of *what the dialect means*.
   If the dialect says `1/0 → +Inf` and the LLM expected `NULL`, no
   amount of token rewriting fixes the answer.
2. **The LLM never has to learn the dialect.** It's just emitting ANSI.
   This defeats the entire training goal: we want the model to natively
   speak novel surfaces, not lean on a translator.
3. **Verification asymmetry.** The transpiler has to know the LLM's
   intent to rewrite correctly. If the LLM means "concat" with `+`, the
   transpiler can only guess from operand types — and if it guesses
   wrong, the reward signal is corrupted.
4. **Coverage cliff.** Plan-shape divergences (`QUALIFY`, `WITH TIES`,
   pivot) and silent semantics (null order, divide-by-zero) are not
   reachable from token rewriting at all.

**Verdict:** rejected. Doesn't actually train the model on the dialect;
turns the project into "generate transpilers."

---

### 2.2 Option B — LLM as world model / state machine

**Idea:** instead of a real engine, use a (cheap) LLM as the executor —
prompt it with the dialect's spec card and the candidate SQL, ask it to
"play SQLite" and return rows. The training agent gets feedback from a
second LLM acting as the environment.

Variant: a state-machine prompt that walks the LLM through "parse →
plan → execute → return rows."

**Why it fails:**

1. **Stochastic feedback breaks RL.** GRPO / RLVR require a deterministic,
   reproducible reward function. If the same SQL on the same data gives
   different rows across calls, the policy gradient is noise.
2. **LLM-as-judge is unreliable for SQL execution.** Hallucinated rows,
   forgotten `WHERE` clauses, made-up join cardinalities. Empirically:
   even GPT-4-class models can't faithfully execute a 4-table join on
   100 rows of data. The error rate is many percent — and the *errors
   correlate* with the agent's own systematic mistakes, which is the
   worst possible bias.
3. **Cost compounds.** Every step of every episode is now an LLM call.
   1000 training tasks × 5 turns × 8 group rollouts = 40k extra calls
   per training step.
4. **No ground truth.** We're training one model against the biases of
   another. There is no fixed point we converge toward; the "correct
   answer" drifts with the world model.
5. **Doesn't honor section 1.5.** Rel algebra is a *deterministic*
   substrate. Replacing it with a probabilistic interpreter throws away
   the structure that makes verification cheap.

**Verdict:** rejected. Training signal would be dominated by world-model
noise, not policy quality.

---

### 2.3 Option C — Real engines for real dialects (Postgres / Snowflake / …)

**Idea:** spin up actual database engines via Docker / drivers / accounts,
one per real-world dialect. Run candidate SQL against the engine; trust
its result.

**Why it fails:**

1. **Doesn't extend beyond what exists.** The whole point is *synthetic*
   dialects — we want hundreds of variants the model has never seen.
   You can't `docker run -it OracleClone47`.
2. **SemanticConfig divergences aren't independently controllable on
   real engines.** You can't ask Postgres to behave like Snowflake on
   one knob and like SQLite on another. Real engines pin the whole
   tuple of knobs at once.
3. **Operational weight.** Drivers, network, credentials, version
   pinning, flaky CI, schema-creation overhead per task. RL needs
   thousands of executions per minute; real engines deliver tens.
4. **Closed scope.** Even with N real engines, the *catalog* of
   divergences we can train on is whatever those engines happen to
   exhibit — not what we want to teach.

**Verdict:** rejected as the primary strategy. Real engines remain
*useful* as cross-checking oracles (DuckDB / SQLite oracles are part of
the verification harness), but they aren't the runtime under test.

---

### 2.4 Option D — One hand-written engine per dialect

**Idea:** for each dialect, write a small Python interpreter that parses
and executes its surface natively. Faithful but bespoke.

**Why it fails:**

1. **Doesn't scale.** Every new dialect is days of engineering. Fine
   for 3 dialects; impossible for 300.
2. **Bug-for-bug reproduction.** Each engine has its own corners,
   discoverable only by running queries against it. We'd be re-creating
   the entire dialect-divergence problem inside our own codebase.
3. **No cross-dialect differential testing.** The same logical query
   has a different *plan representation* in every engine, so we can't
   say "these two dialects should disagree exactly here."
4. **Doesn't honor section 1.5.** If rel algebra is the substrate, the
   N forks share 95% of their code. Forking N times is the wrong
   abstraction.

**Verdict:** rejected. Right answer for a database company; wrong
answer for a synthetic-dialect generator.

---

### 2.5 Option E — Generic query engine + deterministic knobs + LLM lanes (chosen)

The architecture:

```
DialectSpec (Pydantic)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ codegen pipeline:                                                │
│   deterministic emitters (templated from the spec)               │
│       └─► LLM refine loop (only when templates can't express)    │
│                                                                  │
│   emits a dialect package on disk:                               │
│     grammar.lark      semantics.json    overrides.py             │
│     lowering.py       passes.py         effects.py               │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ shared runtime (one copy, scope-locked):                         │
│   IR  ──►  Polars/PyArrow executor parameterized by              │
│           SemanticConfig + per-dialect overrides/passes/effects  │
│   oracle harness (DuckDB / SQLite / reference interpreter /      │
│           property / cross-dialect differential)                 │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
   eval (NL→SQL benchmark) + RL env (GRPO over synthetic dialects)
```

Two extension surfaces, by design:

| Surface | World | Lives in | Covers |
|---|---|---|---|
| `SurfaceSpec` (lexical/syntactic knobs) | **closed** enum | `manysql/spec/dialect.py` | Section 1.1 lex/syn axes |
| `SemanticConfig` (runtime knobs) | **closed** enum | `manysql/spec/semantics.py` | Section 1.3 catalog |
| `lowering.py` (parse-tree → IR) | **open** Python | `manysql/dialects/<name>/` | Surface shapes the templates can't express |
| `overrides.py` (function/operator bodies) | **open** Python | `manysql/dialects/<name>/` | Novel funcs (NVL, MID, …) |
| `passes.py` (Plan-rewrite passes) | **open** Python | `manysql/dialects/<name>/` | Plan-shape sugar (`WITH TIES`, `QUALIFY`) |
| `effects.py` (executor decision points) | **open** Python (closed registry) | `manysql/dialects/<name>/` | Collations, locale-sensitive comparison |

---

### 2.6 Why this option dominates each rejected one

| Concern | Option A (transpile) | Option B (LLM world model) | Option C (real engines) | Option D (per-dialect engines) | **Option E (chosen)** |
|---|---|---|---|---|---|
| Model has to learn the dialect | ✗ | ~ | ✓ | ✓ | **✓** |
| Deterministic feedback | ✓ | ✗ | ✓ | ✓ | **✓** |
| Scales to hundreds of synthetic dialects | ✓ | ✓ | ✗ | ✗ | **✓** |
| Per-knob independent control | ✗ | ~ | ✗ | ✓ | **✓** |
| Cheap per-execution (RL-friendly) | ✓ | ✗ | ✗ | ✓ | **✓** |
| Cross-dialect differential testing | ✗ | ✗ | ~ | ✗ | **✓** |
| Long-tail extensibility | ✗ | ✓ | ✗ | ✓ | **✓** |
| One executor, one IR | n/a | n/a | ✗ | ✗ | **✓** |

---

### 2.7 The decision rule for "where does feature X live"

This is the rule a dialect author follows when adding a divergence,
straight from `manysql/ir/SCOPE.md`:

1. **Pure surface** (different keyword, different operator spelling) →
   `SurfaceSpec` knob.
2. **Runtime divergence with a small enum value space** →
   `SemanticConfig` knob.
3. **Canonical IR is the right *shape*, but the surface produces a
   non-canonical marker** → desugar in `passes.py`.
4. **Canonical IR is the right *shape*, but the executor's
   *implementation* of one op differs** → `effects.py` handler at a
   registered decision point.
5. **Canonical IR cannot represent the feature at all** → Tier-B IR
   extension via RFC.

This rule is the entire policy for "is this a knob or a code lane?"
We re-derive each layer from a property of SQL in section 1:

- Rule 1 ⟵ section 1.1 (lex/syn layers vary independently).
- Rule 2 ⟵ section 1.3 (catalog is small + enumerable).
- Rules 3–4 ⟵ section 1.4 (long tail is open-world).
- Rule 5 ⟵ section 1.6 (scope boundary; IR additions are RFC-gated).

---

### 2.8 Why "deterministic emitter first, LLM refine second"

Not every dialect needs an LLM. Most surface knobs (keyword renames,
operator aliases, comment style) are mechanically templatable from the
spec. The deterministic emitter:

- runs in milliseconds,
- is bit-reproducible across runs,
- doesn't burn tokens / dollars / latency on dialects where templates
  suffice (`mild_postgres_ish`, `snowflake_clone`, `sqlite_clone`,
  `postgres_clone`).

The **LLM refine loop** kicks in only for dialects whose surface needs
structural changes the templates can't express — e.g.
`aggressive_alien` (NIL nulls, `~=` null-safe equality, `OFFSET … FETCH`,
no `ILIKE`, `HAVE` instead of `HAVING`, `ORDERED BY` instead of
`ORDER BY`).

LLM output is **rolled back automatically** if it regresses either the
parse battery or the IR-equivalence battery — the deterministic baseline
is always the fallback. This makes the LLM lane *strictly additive*:
worst case, a generated dialect is identical to its deterministic
baseline.

> **The pattern:** deterministic where deterministic works; LLM only
> where it has to. This is the same pattern as closed-world knobs +
> open-world lanes, applied one level up to *codegen itself*.

---

### 2.9 Section recap — the chosen architecture

- **One IR** (closed, Tier-A relational algebra) — because section 1.5.
- **One executor** parameterized by `SemanticConfig` — because section
  1.3 says the divergence catalog is enumerable.
- **One parser per dialect** (`grammar.lark` + `lowering.py`) — because
  section 1.1 says surfaces vary widely but lower to the same IR.
- **Four open-world Python lanes** for the long tail — because section
  1.4 says some divergence isn't enumerable.
- **Hybrid emitter** (deterministic baseline + LLM refine, with
  rollback) — because most dialects don't need an LLM, and the ones that
  do need verification.
- **Scope-locked** to Tier-A SQL — because section 1.6 says trying to
  generalize past SQL would balloon every other component.

What we *don't* do, and why:

- We don't transpile (can't capture semantic divergence).
- We don't use LLM-as-judge / world-model executors (non-deterministic,
  biased, expensive).
- We don't fork N engines (doesn't scale; defeats synthetic dialects).
- We don't add IR nodes casually (RFC-gated; protects every downstream
  oracle and dialect).

---

*(Sections 3+ will cover: the codegen pipeline in detail, the multi-oracle
verification harness, the RL environment, the eval benchmark, and
results — but those build on this foundation.)*
