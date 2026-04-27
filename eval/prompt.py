"""System-prompt builder.

The original tinybirdco prompt is heavy on ClickHouse / Tinybird specifics
(QUALIFY, AggregateFunction, Tornado templating, multi-node pipes, ...).
Here we render a leaner, dialect-aware version: the executor reports its
schema and dialect label, and we splice them into a stable shell.

Two prompt modes are supported:

* ``plain`` (default) -- model returns the bare SQL with no markdown / no
  tags. Best for closed-source frontier models (GPT-4o, Claude, etc.)
  that follow instructions out of the box.
* ``tag`` -- model wraps the SQL between ``<SQL>`` and ``</SQL>``. This
  is what ``train/grpo_sql.py`` trains for (see
  ``train.env.trl.TRL_TAG_BASE_RULES``); use this mode when evaluating
  any LoRA produced by that training pipeline, otherwise the trained
  model emits the tags anyway and the executor chokes on the raw
  ``<SQL>...`` text. ``extract_sql`` is tag-aware regardless of mode,
  so a trained model that auto-emits tags is still scored correctly
  if ``plain`` is forced -- the mode mostly affects whether the system
  prompt *contradicts* what the model learned.
"""

from __future__ import annotations

import re
from textwrap import dedent
from typing import Literal

from eval.executors.base import SqlExecutor

PromptMode = Literal["plain", "tag"]

# Recognized SQL tag for trained-LoRA-style outputs; matches
# ``train.env.trl._SQL_TAG_RE`` so eval and reward extraction agree.
_SQL_TAG_RE = re.compile(r"<\s*SQL\s*>(.*?)<\s*/\s*SQL\s*>", re.DOTALL | re.IGNORECASE)

_BASE_RULES_PLAIN = """\
- You will be given a question about the data in the database, and a schema.
- Return ONLY the SQL query, with no markdown, no fences, no commentary.
- Generate exactly one SELECT statement (or one WITH ... SELECT). No DDL/DML.
- Add LIMIT to the query when the result could be unbounded; default LIMIT 10.
- Aliases come AFTER the column expression, e.g. `count(*) AS n`.
- Reference the table by its bare name, e.g. `FROM github_events`.
- Use only columns that appear in the schema. Do NOT invent columns.
"""

# Mirrors train.env.trl.TRL_TAG_BASE_RULES so a LoRA trained on that
# instruction sees a familiar wrapper at eval time. In particular: the
# "last <SQL>...</SQL> block wins" rule must match training, otherwise
# a CoT-style model that drafts SQL inside its reasoning gets scored
# against the wrong tag.
_BASE_RULES_TAG = """\
- You will be given a question about the data in the database, and a schema.
- You may reason briefly first if it helps: pick the right tables / joins /
  filters, note edge cases, sketch a draft. Keep the reasoning concise.
- End your reply with one SELECT (or WITH ... SELECT) wrapped in
  <SQL>...</SQL> tags. ONLY THE LAST <SQL>...</SQL> BLOCK in your reply
  is evaluated as your final answer; earlier <SQL>...</SQL> blocks inside
  your reasoning are ignored, so you can draft and revise without penalty.
- Example:
    First I need to count rows in github_events.
    <SQL>SELECT count(*) AS n FROM github_events LIMIT 10</SQL>
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


def build_system_prompt(
    executor: SqlExecutor,
    *,
    prompt_mode: PromptMode = "plain",
) -> str:
    """Compose the system prompt for the given backend.

    ``prompt_mode='tag'`` switches the output-format instructions to
    request ``<SQL>...</SQL>`` wrapping (matching what
    ``train/grpo_sql.py`` trains on). The dialect-specific hints and
    schema block are unchanged across modes.
    """
    dialect = executor.dialect_label().lower()
    hint_key = next(
        (k for k in _DIALECT_HINTS if k in dialect),
        None,
    )
    hint = _DIALECT_HINTS.get(hint_key or "", "")

    rules = _BASE_RULES_TAG if prompt_mode == "tag" else _BASE_RULES_PLAIN

    return dedent(
        f"""\
        You are a careful SQL author. Write a SQL query that answers the user's question.

        {rules.rstrip()}
        {hint.rstrip()}

        Schema:
        {executor.schema_prompt().rstrip()}
        """
    )


def extract_sql(text: str) -> str:
    """Pull the SQL out of an LLM response.

    Order of preference:

    1. ``<SQL>...</SQL>`` tag -- matches what ``train/grpo_sql.py`` trains
       LoRAs to emit. We accept the last tag in the response (so a model
       that thinks-out-loud and then emits the final query in a tag is
       handled correctly), case-insensitive.
    2. Markdown code fence (``\\`\\`\\`sql\\n...\\n\\`\\`\\``) -- frontier
       models sometimes wrap output in fences despite the prompt.
    3. Bare text -- fall through to a strip + semicolon trim.

    The tag-aware path runs *regardless* of which prompt mode was used,
    so a tag-trained LoRA still works even when run with the plain
    prompt (the model emits tags from training; we strip them here).
    """
    if not text:
        return ""

    # 1. <SQL>...</SQL> wins outright when present. We take the last
    # match because some models emit a draft inside a thinking block
    # before the final answer.
    matches = _SQL_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip().rstrip(";").strip()

    # 2/3. Strip code fences then trailing semicolons.
    s = text.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[: -len("```")]
    return s.strip().rstrip(";").strip()


__all__ = ["PromptMode", "build_system_prompt", "extract_sql"]
