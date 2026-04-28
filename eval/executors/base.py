"""SQL executor abstraction.

A backend takes raw SQL text and returns rows (or an error). It also exposes
its schema as a free-form text blob so the system prompt can include it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eval.dataset.questions import Question


@dataclass
class ExecResult:
    """Outcome of running one SQL query against a backend."""

    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    execution_time_s: float = 0.0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rows": self.rows,
            "columns": self.columns,
            "error": self.error,
            "execution_time_s": self.execution_time_s,
            "backend": self.backend,
        }


class SqlExecutor(ABC):
    """Abstract SQL execution backend.

    Lifecycle: `setup()` once per process, `execute()` per query,
    `teardown()` on shutdown. Backends are expected to be thread-unsafe;
    the runner serializes calls.
    """

    name: str = "base"

    @abstractmethod
    def setup(self) -> None:
        """Prepare the backend (build/load datasets, open connections)."""

    @abstractmethod
    def execute(self, sql: str, *, question: Question | None = None) -> ExecResult:
        """Run one query and return rows or an error.

        ``question`` is the originating :class:`Question` (when available);
        executors with a global schema (the default) ignore it. Executors
        whose schema is per-question -- e.g. BIRD where each question
        targets a different ``.sqlite`` file -- use it (typically
        ``question.db_path``) to pick the right backing store. Passing
        ``None`` is always legal and matches the default backend behavior.
        """

    @abstractmethod
    def schema_prompt(self) -> str:
        """Schema description injected into the LLM system prompt."""

    @abstractmethod
    def dialect_label(self) -> str:
        """Short human-friendly dialect name, e.g. 'sqlite' or 'clickhouse'."""

    def teardown(self) -> None:  # pragma: no cover - default no-op
        """Optional cleanup hook."""
        return None


__all__ = ["ExecResult", "SqlExecutor"]
