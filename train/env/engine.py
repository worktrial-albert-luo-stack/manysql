"""Dialect runtime: parse + lower + execute, parametrized over a catalog.

A ``DialectRuntime`` is the layer between "the agent emitted SQL text"
and "we have rows to score". It owns:

* a loaded ``DialectEngine`` (grammar / lowering / semantics / overrides
  / passes / effects), via ``manysql.dialects.DialectRegistry``;
* a Lark parser built from the dialect's grammar;
* a ``CatalogSnapshot`` (tables + IR schemas + a schema prompt).

Lifecycle: construct, call ``setup()`` once, call ``run(sql)`` per query,
optionally call ``teardown()``. Designed to be reused across many tasks
so the parser/grammar build happens exactly once - that's the expensive
part.

The ``run()`` return shape (:class:`~eval.executors.base.ExecResult`) is
deliberately the same one the eval backends return so downstream code
(reward functions, validators, transcript renderers) doesn't need a
second branch for "is this from the env or from eval?".

Error classification on failure (``error_class``) tags whether the
exception came from the parser, the lowering, the executor, or
something else. Reward functions can use the tag to weight a
syntactically-broken response differently from a runtime error like
"unknown column".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING

from eval.executors.base import ExecResult
from manysql.dialects.card import render_dialect_card

if TYPE_CHECKING:
    from lark import Lark

    from manysql.dialects.registry import DialectEngine
    from train.env.catalog import CatalogProvider, CatalogSnapshot


@dataclass
class RunResult:
    """``run()`` return: the eval-style ExecResult plus an error tag."""

    exec_result: ExecResult
    error_class: str | None  # 'parse' | 'runtime' | 'empty' | None on success


class DialectRuntime:
    """Owns the (engine, parser, catalog) triple for one dialect.

    Construction is cheap; ``setup()`` does the heavy work (loading the
    dialect package off disk, building the Lark parser, materializing
    the catalog). The runtime is single-threaded; spin up one per
    worker thread if you parallelize.
    """

    def __init__(self, *, dialect: str, catalog: CatalogProvider) -> None:
        self.dialect = dialect
        self.catalog_provider = catalog
        self._engine: DialectEngine | None = None
        self._parser: Lark | None = None
        self._snapshot: CatalogSnapshot | None = None
        self._dialect_card: str = ""
        # Map of Lark anonymous-terminal names (`__ANON_3`) to their literal
        # surface form (``<>``, ``<=``, ``||``...). Built once at setup so we
        # can humanize parse errors before showing them to the model -- raw
        # ``__ANON_3`` is opaque, ``<>`` is actionable.
        self._anon_terminal_map: dict[str, str] = {}

    def setup(self) -> None:
        from lark import Lark, LarkError  # noqa: PLC0415

        from manysql.dialects.registry import DialectRegistry  # noqa: PLC0415

        engine = DialectRegistry().load(self.dialect)
        try:
            parser = Lark(engine.grammar_text, start="start", parser="earley")
        except LarkError as exc:
            raise RuntimeError(
                f"failed to build parser for dialect {self.dialect!r}: {exc}"
            ) from exc

        self._engine = engine
        self._parser = parser
        self._snapshot = self.catalog_provider.build()
        self._dialect_card = render_dialect_card(engine)
        self._anon_terminal_map = _build_anon_terminal_map(parser)

    def teardown(self) -> None:
        self._engine = None
        self._parser = None
        self._snapshot = None
        self._dialect_card = ""
        self._anon_terminal_map = {}

    @property
    def engine(self) -> DialectEngine:
        if self._engine is None:
            raise RuntimeError("DialectRuntime.setup() was not called")
        return self._engine

    @property
    def snapshot(self) -> CatalogSnapshot:
        if self._snapshot is None:
            raise RuntimeError("DialectRuntime.setup() was not called")
        return self._snapshot

    @property
    def dialect_card(self) -> str:
        return self._dialect_card

    @property
    def schema_prompt(self) -> str:
        return self.snapshot.schema_prompt

    # -------- system prompt composition --------

    def system_prompt(self, *, base_rules: str | None = None) -> str:
        """Compose the LLM system prompt: base rules + dialect card + schema."""
        rules = (base_rules if base_rules is not None else _DEFAULT_RULES).rstrip()
        card = self._dialect_card.rstrip()
        schema = self.snapshot.schema_prompt.rstrip()
        return dedent(
            f"""\
            You are a careful SQL author. Write a SQL query that answers the user's question.

            {rules}

            {card}

            Schema:
            {schema}
            """
        )

    # -------- query execution --------

    def run(self, sql: str) -> RunResult:
        """Parse + lower + execute one query. Never raises."""
        from manysql.executor import execute as plan_execute  # noqa: PLC0415

        if self._engine is None or self._parser is None or self._snapshot is None:
            raise RuntimeError("DialectRuntime.setup() was not called")

        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return RunResult(
                exec_result=ExecResult(
                    success=False,
                    error="empty SQL",
                    backend=f"manysql:{self.dialect}",
                ),
                error_class="empty",
            )

        start = time.perf_counter()
        try:
            tree = self._parser.parse(sql)
        except Exception as exc:  # Lark raises a number of subclasses; tag them all 'parse'.
            msg = _humanize_parse_error(
                f"{type(exc).__name__}: {exc}", self._anon_terminal_map
            )
            return RunResult(
                exec_result=ExecResult(
                    success=False,
                    error=msg,
                    execution_time_s=time.perf_counter() - start,
                    backend=f"manysql:{self.dialect}",
                ),
                error_class="parse",
            )

        try:
            plan = self._engine.lowering.lower(
                tree, self._engine.semantics, self._snapshot.schemas
            )
            df = plan_execute(
                plan,
                self._engine.semantics,
                self._snapshot.tables,
                overrides=self._engine.overrides,
                passes=self._engine.passes,
                effects=self._engine.effects,
            )
        except Exception as exc:
            # Lowering and execution share the same error lane; both manifest
            # as "the SQL parsed but means something the engine can't run".
            return RunResult(
                exec_result=ExecResult(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    execution_time_s=time.perf_counter() - start,
                    backend=f"manysql:{self.dialect}",
                ),
                error_class="runtime",
            )

        return RunResult(
            exec_result=ExecResult(
                success=True,
                rows=df.to_dicts(),
                columns=list(df.columns),
                execution_time_s=time.perf_counter() - start,
                backend=f"manysql:{self.dialect}",
            ),
            error_class=None,
        )

    def __enter__(self) -> DialectRuntime:
        self.setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.teardown()


# ---------------------------------------------------------------------------
# Parse-error humanization
# ---------------------------------------------------------------------------
#
# Lark's parse errors include an "Expected one of: ... __ANON_N ..." list,
# where __ANON_N is the auto-generated name for any terminal that was defined
# inline as a string/regex literal in the grammar (operators like ``<>``,
# ``<=``, ``||``, plus a handful of case-insensitive keyword regexes). Those
# names are meaningless to a model trying to self-correct -- so at parser
# build we snapshot the name-to-pattern mapping, then rewrite the error
# message before it leaves the runtime.


_ANON_PREFIX = "__ANON_"
_ANON_LINE_RE = re.compile(rf"\* ({re.escape(_ANON_PREFIX)}\d+)")


def _build_anon_terminal_map(parser: Lark) -> dict[str, str]:
    """Snapshot ``{terminal_name: friendly_form}`` for all ``__ANON_*``."""
    out: dict[str, str] = {}
    terminals = getattr(parser, "terminals", None)
    if not terminals:
        return out
    for term in terminals:
        name = getattr(term, "name", "")
        if not name.startswith(_ANON_PREFIX):
            continue
        pattern = getattr(term, "pattern", None)
        raw = getattr(pattern, "value", None) if pattern is not None else None
        if not isinstance(raw, str) or not raw:
            continue
        out[name] = _humanize_terminal_pattern(raw)
    return out


def _humanize_terminal_pattern(raw: str) -> str:
    """Strip Lark's regex wrappers so the pattern reads as the literal SQL form.

    Lark stores string-literal terminals as their plain text (``<>``) and
    regex-literal terminals wrapped in ``(?i:...)`` for case-insensitive
    keyword matches. We unwrap both. Backslash escapes (``\\|``, ``\\(``)
    are removed so ``||`` reads as ``||`` instead of ``\\|\\|``.
    """
    s = raw
    m = re.fullmatch(r"\(\?i:(.+)\)", s)
    if m:
        s = m.group(1)
    s = re.sub(r"\\(.)", r"\1", s)
    return s


def _humanize_parse_error(msg: str, anon_map: dict[str, str]) -> str:
    """Rewrite ``__ANON_N`` mentions in a Lark error message to readable form.

    Conservative: only touches ``* __ANON_N`` lines (Lark's expected-token
    list format). Anything else passes through unchanged so we don't
    accidentally rewrite a legitimate ``__ANON_5`` substring elsewhere.
    """
    if not anon_map or _ANON_PREFIX not in msg:
        return msg

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        return f"* {anon_map.get(name, name)}"

    return _ANON_LINE_RE.sub(_sub, msg)


# Default base SQL rules. Mirrors ``eval.prompt._BASE_RULES`` but lives
# here so train doesn't depend on eval for prompt-building. If you want
# to share, pass ``base_rules=`` explicitly to ``system_prompt()``.
_DEFAULT_RULES = """\
- You will be given a question about the data in the database, and a schema.
- Return ONLY the SQL query, with no markdown, no fences, no commentary.
- Generate exactly one SELECT statement (or one WITH ... SELECT). No DDL/DML.
- Add LIMIT to the query when the result could be unbounded; default LIMIT 10.
- Aliases come AFTER the column expression, e.g. `count(*) AS n`.
- Reference each table by its bare name. Do not invent columns.
"""


__all__ = ["DialectRuntime", "RunResult"]
