"""Lowering stub for _test_ci_eq.

The dialect tests the effects lane and doesn't have a SQL surface that
gets parsed in tests. `lower(...)` raises immediately; tests construct
IR plans directly and execute them with `effects=engine.effects`.
"""

from __future__ import annotations

from typing import Any

from manysql.ir.plan import Plan


def lower(tree: Any, config: Any, catalog: Any) -> Plan:  # pragma: no cover - stub
    raise NotImplementedError(
        "_test_ci_eq has no SQL surface; effects are exercised directly "
        "from test fixtures."
    )
