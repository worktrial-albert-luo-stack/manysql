"""Golden query corpus.

This is the canonical correctness corpus. Every query here:
1. Parses through the reference dialect's grammar.
2. Lowers to a valid IR Plan.
3. Executes through the Polars executor.
4. Agrees across all applicable oracles via the harness.

When validating a generated dialect, we re-render each query into that
dialect's surface (when feasible) or skip with `cross_dialect=False`.
"""

from manysql.golden.queries import GOLDEN_QUERIES, GoldenQuery

__all__ = ["GOLDEN_QUERIES", "GoldenQuery"]
