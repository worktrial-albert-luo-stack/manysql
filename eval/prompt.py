"""System-prompt builder.

The original tinybirdco prompt is heavy on ClickHouse / Tinybird specifics
(QUALIFY, AggregateFunction, Tornado templating, multi-node pipes, ...).
Here we render a leaner, dialect-aware version: the executor reports its
schema and dialect label, and we splice them into a stable shell.
"""

from __future__ import annotations

from textwrap import dedent

from eval.executors.base import SqlExecutor

_BASE_RULES = """\
- You will be given a question about the data in the database, and a schema.
- Return ONLY the SQL query, with no markdown, no fences, no commentary.
- Generate exactly one SELECT statement (or one WITH ... SELECT). No DDL/DML.
- Add LIMIT to the query when the result could be unbounded; default LIMIT 10.
- Aliases come AFTER the column expression, e.g. `count(*) AS n`.
- Reference the table by its bare name, e.g. `FROM github_events`.
- Use only columns that appear in the schema. Do NOT invent columns.
"""


_DIALECT_HINTS: dict[str, str] = {
    "sqlite": dedent(
        """\
        - Target dialect: SQLite. Use SQLite-flavored SQL.
        - Datetime columns are TEXT in ISO 8601 ('YYYY-MM-DD HH:MM:SS'). Use
          `strftime('%Y-%m', col)`, `strftime('%Y', col)`, `date(col)`, `julianday(col)` etc.
        - SQLite has no boolean type; use 0/1 integers.
        - SQLite has no native array type. Array-shaped columns (like `labels`)
          are comma-separated strings. To check membership, use
          `',' || labels || ',' LIKE '%,bug,%'`.
        - For COUNT use `count(*)`, not `count()`.
        - Use `group_concat`, not `array_agg` / `LISTAGG`.
        - String concatenation is `||`, not `CONCAT(...)`.
        """
    ),
    "clickhouse": dedent(
        """\
        - Target dialect: ClickHouse (Tinybird).
        - Use ClickHouse functions (count(), toStartOfMonth, splitByChar, has(arr, x), etc.).
        - Do NOT use CTEs / WITH clauses (Tinybird recommends node reuse instead).
        """
    ),
    "manysql:": dedent(
        """\
        - Target dialect: a manysql synthetic dialect, executed by the
          manysql IR engine. The schema block below begins with a
          'dialect card' that enumerates EVERY surface and semantic
          divergence from a near-ANSI baseline. Read it carefully and
          use ONLY the surface forms it (or the baseline) describes.
        - The dialect's grammar is strict; unknown keywords, operators,
          or function spellings will be rejected at parse time.
        - When the card lists keyword / operator / cast / limit
          alternatives, use the listed spelling, not its ANSI form.
        - Array-shaped columns (e.g. `labels`) are comma-separated TEXT.
          Match with `LIKE '%,bug,%'` patterns; do not call array_*
          functions unless the dialect card lists them as aliases.
        - For dates (ISO-8601 TEXT) prefer SUBSTR(col, 1, 4) /
          SUBSTR(col, 1, 7); strftime is not guaranteed in synthetic
          dialects.
        """
    ),
}


def build_system_prompt(executor: SqlExecutor) -> str:
    """Compose the system prompt for the given backend."""
    dialect = executor.dialect_label().lower()
    hint_key = next(
        (k for k in _DIALECT_HINTS if k in dialect),
        None,
    )
    hint = _DIALECT_HINTS.get(hint_key or "", "")

    return dedent(
        f"""\
        You are a careful SQL author. Write a SQL query that answers the user's question.

        {_BASE_RULES.rstrip()}
        {hint.rstrip()}

        Schema:
        {executor.schema_prompt().rstrip()}
        """
    )


def extract_sql(text: str) -> str:
    """Strip code fences and trailing semicolons that LLMs occasionally emit."""
    s = text.strip()
    if s.startswith("```"):
        # remove opening fence (optionally with language)
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[: -len("```")]
    return s.strip().rstrip(";").strip()


__all__ = ["build_system_prompt", "extract_sql"]
