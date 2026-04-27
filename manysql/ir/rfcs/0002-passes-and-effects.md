# RFC 0002 — Per-dialect extension lanes: `passes.py` and `effects.py`

Status: accepted \
Author: manysql core \
Targets: v1 (lands alongside the Tier-A IR lock)

## Summary

manysql v1 generates dialects through a hybrid pipeline: closed-world
`SurfaceSpec` and `SemanticConfig` knobs cover the bulk of practical
divergence, and an open-world `lowering.py` (parse-tree → IR) plus
`overrides.py` (function / operator bodies) handle dialect-specific
syntax and special-cased computations.

Two lanes were missing from this story:

1. A way to describe *plan-shape* divergence — features whose surface
   produces IR the canonical executor doesn't understand directly, but
   whose intent is fully expressible in canonical IR.
2. A way to describe *open-ended runtime divergence* — operator
   semantics that vary along an axis whose value space is too large to
   enumerate as a `SemanticConfig` enum (collations, locale-sensitive
   comparison, custom rounding modes, etc.).

This RFC introduces two new per-dialect lanes that fill those gaps
without growing the IR or the closed knobs:

- **`passes.py`** — Plan rewrites that run between lowering and
  execution.
- **`effects.py`** — Named handlers swapped into specific executor
  decision points.

Together with `lowering.py` and `overrides.py` they form the four
extension lanes catalogued in `SCOPE.md`. None of them mutate the IR.

## Goals

1. Keep the IR (`manysql/ir/`) closed and additive-only (Tier A / Tier
   B contract).
2. Let a dialect ship features that need either a different *plan
   shape* or a different *implementation* of a canonical op without
   editing the executor.
3. Keep both lanes optional: a dialect that needs neither pays no cost
   and the codegen pipeline emits empty stubs.
4. Stay verification-friendly: every lane that runs at execution time
   must be visible to the oracle harness so the same `polars_execute`
   call observed by the harness is the one a real client sees.

## Non-goals

- A free-form hook surface inside the executor. Effects must register
  against a fixed name; ad-hoc executor mutation belongs in `lowering.py`
  / `passes.py`.
- A way to add new IR nodes from a dialect. That is still the
  Tier-B IR-extension RFC process.
- A way to influence oracles. Oracles run reference engines; lane
  modules apply only to the manysql executor under test.

## Lane 1 — `passes.py` (Plan-rewrite passes)

### Contract

A dialect's `passes.py` exposes:

```python
PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]]
```

Each callable accepts the IR plan and the active `SemanticConfig` and
returns a (possibly rewritten) plan. They run in list order between the
dialect's `lowering.lower(...)` step and the canonical executor's
dispatch. The empty list (the deterministic emitter's default) is a
no-op.

A pass that returns `None` is a bug; the runtime raises `RuntimeError`
so dialect authors fail fast rather than silently dropping the plan.

### When to use it

When a dialect's surface produces IR the canonical executor doesn't
understand directly, but the *intent* of that IR is fully expressible
in canonical IR. The pass is the desugaring step.

Worked example: `LIMIT N WITH TIES` (T-SQL `TOP N WITH TIES`).

The dialect's `lowering.py` emits a sentinel marker — a `Filter` whose
predicate is `FuncCall("__manysql_with_ties", [Literal(N)])` wrapping a
`Sort`. The canonical executor would raise on the `FuncCall`. The pass
recognizes the pattern and rewrites it to canonical IR:

```
Project(
  input=Filter(
    input=Window(  # rank() over (order by <Sort.keys>)
      input=<original Sort>,
      windows=[("__manysql_rk", WindowCall(RANK, order_by=<keys>))]
    ),
    predicate=BinaryOp(LTE, ColumnRef("__manysql_rk"), Literal(N))
  ),
  projections=<original schema columns mapped through unchanged>
)
```

The proof point lives in `manysql/dialects/_test_with_ties/` and
`tests/test_dialect_passes.py`.

### When NOT to use it

- For *runtime* divergence (e.g. case-insensitive equality). That is
  what `effects.py` exists for.
- For features whose canonical IR doesn't exist. Those need a Tier-B
  IR extension.
- For function-body or operator-body overrides. Those belong in
  `overrides.py`.

### Engine integration

`PlanExecutor.execute(plan)` calls `apply_pre_passes(plan, semantics,
self.passes)` once at the top of dispatch. The helper is exported from
`manysql.executor.engine` so the oracle harness can also apply passes
when materializing `actual` itself.

A common shape is "rewrite-children-then-rewrite-self" so a marker
appearing anywhere in the plan tree is desugared. The synthetic
`_test_with_ties` dialect ships an explicit `_map_children` walker per
Plan subtype — the IR is closed, so this is small and stable code.

### Verification

Passes are observable to the oracle harness by construction: when
`OracleHarness.verify(plan, semantics, catalog, passes=…)` is called
without an `actual`, the harness routes its `polars_execute` through
the same passes the runtime would apply. Inter-oracle comparison still
runs on the *original* plan because oracles are reference engines, not
the system under test; their job is to evaluate what the dialect *means*,
which is the canonical-IR version.

## Lane 2 — `effects.py` (Executor effects)

### Contract

A dialect's `effects.py` exposes:

```python
EFFECTS: dict[str, Callable]
```

Each key is a registered effect name; each value is a callable whose
signature is fixed for that name. When an effect name is absent or
returns `None` the executor falls back to its canonical implementation.

Effects are looked up on the per-call path through `ExprEvaluator`
(and, when relevant, `PlanExecutor`); the lookup is cheap (a `getattr`
plus a dict probe) so dialects that ship no effects pay essentially
nothing.

### v1 effect registry

| Name | Signature | Replaces |
|---|---|---|
| `text_eq` | `(left: pl.Expr, right: pl.Expr, semantics: SemanticConfig) -> Optional[pl.Expr]` | `Op.EQ` Polars implementation in `ExprEvaluator._binary` |
| `text_neq` | `(left: pl.Expr, right: pl.Expr, semantics: SemanticConfig) -> Optional[pl.Expr]` | `Op.NEQ` Polars implementation |
| `text_in_pattern` | `(operand: pl.Expr, pattern: str, semantics: SemanticConfig, case_sensitive: bool) -> Optional[pl.Expr]` | `Op.LIKE` / `Op.ILIKE` pattern dispatch |

Returning `None` from a handler explicitly defers to the canonical
implementation. This lets a dialect target only a sub-domain (e.g.
collation-insensitive comparison only when both operands are
text-typed) without rewriting everything.

### When to use it

When a canonical IR shape is exactly what the dialect wants, but the
*implementation* of one operation has to differ for this dialect. The
effect swaps the body without changing the IR.

Worked example: T-SQL's default `Latin1_General_CI_AS` collation. The
plan IR for `WHERE name = 'alice'` is identical to the reference
dialect's; only the `=` evaluator changes. The synthetic
`_test_ci_eq` dialect installs a `text_eq` effect that lowercases both
sides:

```python
def _ci_eq(left, right, semantics):
    return left.cast(pl.Utf8).str.to_lowercase() == right.cast(pl.Utf8).str.to_lowercase()

EFFECTS = {"text_eq": _ci_eq}
```

The proof point lives in `manysql/dialects/_test_ci_eq/` and
`tests/test_dialect_effects.py`.

### When NOT to use it

- For closed-world semantic knobs (null ordering, divide-by-zero,
  COUNT-on-empty, …). Those belong in `SemanticConfig`.
- For feature *bodies* the canonical executor doesn't recognize at all
  (e.g. a custom function name). Those belong in `overrides.py`.
- For plan-shape divergence. That is `passes.py`.

### Adding a new effect to the registry

The `EFFECTS` dict is open in the sense that each dialect picks what to
populate. The *registry* — the set of effect names the executor
consults — is closed and grows via this RFC process. Adding a new
effect requires:

1. Pick a name and a signature. The signature must accept enough
   IR / Polars context that a dialect handler can decide; it must
   accept `semantics` so handlers can read closed-world knobs they
   compose with.
2. Wire the executor decision point: locate the canonical
   implementation in `ExprEvaluator` / `PlanExecutor`, replace it with
   `eff = self._call_effect(name, …); return eff if eff is not None
   else <canonical>`.
3. Document the new entry in `manysql/codegen/effects_emit.py` (so
   generated stubs reflect the v1 registry) and in this RFC.
4. Add at least one synthetic dialect under `manysql/dialects/_test_*`
   that installs the handler, plus a regression test under
   `tests/test_dialect_effects.py`.
5. Verify that the oracle harness's `actual` materialization forwards
   `effects` (the v1 wiring is already in place; new effects inherit
   it).

The registry is intentionally small in v1 — three entries that cover
the most-requested T-SQL divergence (collation-insensitive comparison,
collation-aware LIKE). New effects land case-by-case as target
dialects motivate them.

## Loader and sandbox

Both lanes are loaded through the same registry path as `lowering.py`
and `overrides.py`: `DialectRegistry.load(name)` calls
`importlib.util.spec_from_file_location` for each present file. The
in-memory sandboxed loader (`manysql/codegen/overrides_loader.py`,
function `load_sandboxed_module`) is used by codegen-time validators
(LLM-output gating); production loads of on-disk dialect packages are
trusted. The allowlist now includes `manysql.ir.expr`, `manysql.ir.plan`,
and `manysql.ir.types` so passes can construct IR nodes and effects
can express IR-typed signatures when they need to.

## Codegen pipeline integration

`PackageBundle` carries `passes_py` and `effects_py` source text; both
are written to the dialect package alongside `overrides.py`. The
deterministic emitters (`manysql/codegen/passes_emit.py`,
`manysql/codegen/effects_emit.py`) produce empty templates today —
spec-aware population lands when a `SurfaceSpec` knob explicitly
demands a pass (e.g. `with_ties: bool`) or a `SemanticDivergences` knob
explicitly demands an effect (e.g. `text_collation`). The lanes
themselves are usable hand-written by dialect authors immediately.

## Migration & compatibility

- Existing dialects (`_reference`, `mild_postgres_ish`, etc.) ship no
  `passes.py` / `effects.py`; the registry treats them as absent and
  the runtime is identical to v1-pre-RFC.
- The codegen pipeline writes empty stubs for all newly-generated
  dialects; loading those stubs is a no-op.
- The `DialectEngine` dataclass grew two optional fields (`passes`,
  `effects`); they default to `None`. Callers that constructed
  `DialectEngine` positionally are not affected because both new
  fields are after the existing optional `overrides`.

## Verification budget

- `tests/test_dialect_passes.py` — 8 tests covering registry load,
  rewrite shape, end-to-end execution, negative control (no passes →
  marker raises), no-op behavior with missing / empty modules, and
  harness forwarding.
- `tests/test_dialect_effects.py` — 6 tests covering registry load,
  default behavior preservation, CI equality, CI inequality, and
  empty-registry fall-through.
- The full pre-existing suite (232 tests) continues to pass; total
  after this RFC is 245.

## Decision log

| Lane | Status | Decision date | Notes |
|------|--------|---------------|-------|
| `passes.py`   | accepted | 2026-04-26 | Lands with v1; proof: `_test_with_ties`. |
| `effects.py`  | accepted | 2026-04-26 | Lands with v1; v1 registry: `text_eq`, `text_neq`, `text_in_pattern`. |
