"""Polars/PyArrow IR executor.

Entry points:
    PlanExecutor(catalog, semantics).execute(plan) -> pl.DataFrame
    execute(plan, semantics, catalog) -> pl.DataFrame
"""

from manysql.executor.engine import PlanExecutor, apply_pre_passes, execute
from manysql.executor.expr_eval import ExprEvaluator

__all__ = ["PlanExecutor", "execute", "ExprEvaluator", "apply_pre_passes"]
