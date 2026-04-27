"""Generated Plan -> Plan rewrites for the postgres_sqlite_server_inside dialect.

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
