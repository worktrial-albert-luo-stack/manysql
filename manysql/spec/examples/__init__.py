"""Worked DialectSpec examples used as fixtures for codegen and testing.

The examples sit on a divergence ladder:
    - `mild_postgres_ish`: identifier-fold, NULL ordering, sum-of-empty=0
    - `moderate_keyword_swap`: SELECT->PICK, FROM->IN, ILIKE removed
    - `aggressive_alien`: invented operators, exotic CAST, prefix wildcard

There are also three real-world clone specs used as eval-harness baselines:
    - `snowflake_clone`: faithful Snowflake (UPPER fold, ILIKE, error on
      div-by-zero, integer division promotes, ``//`` line comments,
      TRY_CAST/NVL/IFNULL aliases).
    - `sqlite_clone`: faithful SQLite (preserve-case ASCII-CI identifiers,
      NULL on div-by-zero, truncating int div, case-insensitive LIKE, no
      ILIKE, C-style truthiness, IFNULL/SUBSTRING aliases).
    - `postgres_clone`: faithful Postgres (lower fold, NULLS LAST on ASC,
      error on div-by-zero, truncating int div, case-sensitive LIKE,
      native ILIKE, strict booleans, CHAR_LENGTH/SUBSTRING aliases).

Each is small enough to read end-to-end and intentionally hits different parts
of the codegen surface so they exercise distinct refine paths.
"""

from manysql.spec.examples.aggressive_alien import AGGRESSIVE_ALIEN
from manysql.spec.examples.mild_postgres_ish import MILD_POSTGRES_ISH
from manysql.spec.examples.moderate_keyword_swap import MODERATE_KEYWORD_SWAP
from manysql.spec.examples.postgres_clone import POSTGRES_CLONE
from manysql.spec.examples.snowflake_clone import SNOWFLAKE_CLONE
from manysql.spec.examples.sqlite_clone import SQLITE_CLONE

EXAMPLE_SPECS = {
    "mild_postgres_ish": MILD_POSTGRES_ISH,
    "moderate_keyword_swap": MODERATE_KEYWORD_SWAP,
    "aggressive_alien": AGGRESSIVE_ALIEN,
    "snowflake_clone": SNOWFLAKE_CLONE,
    "sqlite_clone": SQLITE_CLONE,
    "postgres_clone": POSTGRES_CLONE,
}

__all__ = [
    "MILD_POSTGRES_ISH",
    "MODERATE_KEYWORD_SWAP",
    "AGGRESSIVE_ALIEN",
    "SNOWFLAKE_CLONE",
    "SQLITE_CLONE",
    "POSTGRES_CLONE",
    "EXAMPLE_SPECS",
]
