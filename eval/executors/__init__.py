"""Pluggable SQL execution backends for the eval harness.

Backends share the `SqlExecutor` protocol from `base.py`. Pick one via
`get_executor(name, **kwargs)`.
"""

from eval.executors.base import ExecResult, SqlExecutor
from eval.executors.factory import get_executor

__all__ = ["ExecResult", "SqlExecutor", "get_executor"]
