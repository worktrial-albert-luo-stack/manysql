"""Emit passes.py for a generated dialect.

Passes are `Plan -> Plan` rewrites that run *between* the dialect's
parse-tree-to-IR lowering and the canonical executor. They desugar
non-canonical IR markers (emitted by a dialect's `lowering.py` for
features its surface supports but the canonical IR does not) into
canonical IR shapes the executor already understands.

The contract:

    PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]]

Each pass takes a Plan and the SemanticConfig and returns a (possibly
rewritten) Plan. They run in list order; later passes see the output of
earlier ones. The empty list (the deterministic emitter's default) means
"no rewrites" and the runtime forwards the IR straight to the executor.

The deterministic emitter writes an empty list. LLM-refined emitters or
hand-written passes populate it for dialects whose surface needs it
(e.g. WITH TIES, GROUPING SETS, PIVOT lowered to canonical Aggregate +
CASE).
"""

from __future__ import annotations

from manysql.spec.dialect import DialectSpec


_PASSES_TEMPLATE = '''"""Generated Plan -> Plan rewrites for the {name} dialect.

This module exposes one public list:

    PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]]
        Each callable accepts the IR Plan and the active SemanticConfig
        and returns a (possibly rewritten) Plan. They run in list order
        between the dialect's `lowering.lower(...)` step and the
        canonical executor.

The deterministic codegen writes an empty list. Hand-written or
LLM-refined emitters populate it when the dialect needs to desugar
non-canonical IR markers into canonical shapes the executor handles.
"""

from __future__ import annotations

from typing import Callable

from manysql.ir.plan import Plan
from manysql.spec.semantics import SemanticConfig


PRE_EXECUTION_PASSES: list[Callable[[Plan, SemanticConfig], Plan]] = []
'''


def emit_passes(spec: DialectSpec) -> str:
    """Return the passes.py source for the dialect.

    Currently always returns the empty-list template. Spec-aware
    population is deferred until the codegen pipeline grows knowledge
    of features that *require* a rewrite pass (e.g. a `with_ties`
    surface knob landing in `SurfaceSpec`).
    """
    return _PASSES_TEMPLATE.format(name=spec.name)


__all__ = ["emit_passes"]
