"""Worked DialectSpec examples used as fixtures for codegen and testing.

The examples sit on a divergence ladder:
    - `mild_postgres_ish`: identifier-fold, NULL ordering, sum-of-empty=0
    - `moderate_keyword_swap`: SELECT->PICK, FROM->IN, ILIKE removed
    - `aggressive_alien`: invented operators, exotic CAST, prefix wildcard

Each is small enough to read end-to-end and intentionally hits different parts
of the codegen surface so they exercise distinct refine paths.
"""

from manysql.spec.examples.aggressive_alien import AGGRESSIVE_ALIEN
from manysql.spec.examples.mild_postgres_ish import MILD_POSTGRES_ISH
from manysql.spec.examples.moderate_keyword_swap import MODERATE_KEYWORD_SWAP

EXAMPLE_SPECS = {
    "mild_postgres_ish": MILD_POSTGRES_ISH,
    "moderate_keyword_swap": MODERATE_KEYWORD_SWAP,
    "aggressive_alien": AGGRESSIVE_ALIEN,
}

__all__ = [
    "MILD_POSTGRES_ISH",
    "MODERATE_KEYWORD_SWAP",
    "AGGRESSIVE_ALIEN",
    "EXAMPLE_SPECS",
]
