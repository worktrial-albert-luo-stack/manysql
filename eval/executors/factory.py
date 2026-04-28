"""Factory for picking a backend by name without import-time side effects."""

from __future__ import annotations

from typing import Any

from eval.executors.base import SqlExecutor


def get_executor(name: str, **kwargs: Any) -> SqlExecutor:
    """Resolve a backend name to a constructed (but not yet `setup()`'d) executor.

    Names: 'sqlite' (default), 'tinybird', 'synthetic', 'bird'.

    Imports are deliberately deferred so that selecting `sqlite` doesn't pull
    in tinybird/synthetic/BIRD optional dependencies.
    """
    name = name.lower()
    if name == "sqlite":
        from eval.executors.sqlite_executor import SqliteExecutor  # noqa: PLC0415

        return SqliteExecutor(**kwargs)
    if name == "tinybird":
        from eval.executors.tinybird_executor import TinybirdExecutor  # noqa: PLC0415

        return TinybirdExecutor(**kwargs)
    if name == "synthetic":
        from eval.executors.synthetic_executor import SyntheticExecutor  # noqa: PLC0415

        return SyntheticExecutor(**kwargs)
    if name == "bird":
        from eval.executors.bird_sqlite_executor import (  # noqa: PLC0415
            BirdSqliteExecutor,
        )

        return BirdSqliteExecutor(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Choose one of: sqlite, tinybird, "
        f"synthetic, bird"
    )


__all__ = ["get_executor"]
