"""Oracle layer: multi-truth-source verification.

The harness (manysql/oracle/harness.py) selects the strongest applicable oracle
for each plan and runs others opportunistically. Disagreement -> needs_review.
"""

from manysql.oracle.base import (
    Oracle,
    OracleCapability,
    OracleResult,
    frames_equal,
    is_order_sensitive,
    normalize_for_comparison,
)
from manysql.oracle.cross_dialect import (
    CrossDialectMember,
    CrossDialectOracle,
    CrossDialectReport,
    CrossDialectVerdict,
    DialectExecution,
)
from manysql.oracle.duckdb_oracle import DuckDBOracle
from manysql.oracle.harness import (
    HarnessReport,
    OracleHarness,
    Verdict,
    default_oracles,
    default_property_oracles,
)
from manysql.oracle.property_oracle import PropertyOracle
from manysql.oracle.reference_interpreter import ReferenceInterpreter
from manysql.oracle.sql_render import SqlDialectFlags, UnsupportedByEngine, render_plan
from manysql.oracle.sqlite_oracle import SQLiteOracle

__all__ = [
    "Oracle",
    "OracleCapability",
    "OracleResult",
    "ReferenceInterpreter",
    "DuckDBOracle",
    "SQLiteOracle",
    "PropertyOracle",
    "OracleHarness",
    "HarnessReport",
    "Verdict",
    "default_oracles",
    "default_property_oracles",
    "CrossDialectOracle",
    "CrossDialectMember",
    "CrossDialectReport",
    "CrossDialectVerdict",
    "DialectExecution",
    "render_plan",
    "SqlDialectFlags",
    "UnsupportedByEngine",
    "frames_equal",
    "normalize_for_comparison",
    "is_order_sensitive",
]
