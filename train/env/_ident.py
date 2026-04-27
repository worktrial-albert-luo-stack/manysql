"""Identifier sanitization helpers shared across task generators.

The synthetic dialect grammars accept ``c_<lowercase_alnum>``-style
identifiers everywhere a column or table name is expected. Real-world
datasets (WikiSQL Wikipedia tables, BIRD-SQL Kaggle/competition
schemas) use arbitrary headers full of spaces, parentheses, slashes,
percent signs, diacritics, and non-ASCII glyphs. These helpers map any
input string to a deterministic safe identifier and resolve collisions
that arise when several originals collapse to the same sanitized form.

Used by:

* :mod:`train.env.wikisql` -- 1 table per task, simple column rename.
* :mod:`train.env.bird`    -- multi-table real schemas; ``_safe_table_name``
  also gets db_id-prefixed table names.

Kept dependency-free (stdlib only) so ``train.env`` modules can import
this without pulling polars / lark / datasets.
"""

from __future__ import annotations

import re
import unicodedata


def safe_ident(raw: str, *, fallback: str = "x") -> str:
    """Map an arbitrary string to a safe SQL identifier.

    Strips diacritics, replaces non-alphanumeric runs with underscores,
    lowercases, and prepends ``c_`` so reserved-word collisions are
    avoided regardless of dialect. Empty results fall back to
    ``c_<fallback>`` (e.g. when the original heading was a single
    non-ASCII glyph that gets stripped entirely).
    """
    norm = unicodedata.normalize("NFKD", raw or "")
    norm = norm.encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    if not norm:
        norm = fallback
    return f"c_{norm}"


def safe_table_name(raw: str, *, prefix: str = "", fallback: str = "anon") -> str:
    """Map an arbitrary string to a safe table name.

    Same shape as :func:`safe_ident` but without the ``c_`` prefix --
    callers usually want their own (``wikisql_``, ``bird_<db>__``).
    """
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", raw or "").strip("_").lower()
    if not norm:
        norm = fallback
    return f"{prefix}{norm}" if prefix else norm


def dedupe_columns(names: list[str]) -> list[str]:
    """Resolve duplicate sanitized column names by suffixing _1, _2, ...

    Some real-world tables have two columns whose original headers
    collapse to the same sanitized form (e.g. ``"Score (1)"`` and
    ``"Score (2)"`` both become ``c_score``). Deduping keeps the first
    occurrence and suffixes the rest so the IR schema stays unique.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


__all__ = ["dedupe_columns", "safe_ident", "safe_table_name"]
