"""Card-conformance gate.

The dialect ``card`` is the LLM-facing description of the dialect: the
patterns and worked examples a model is told to use. If the card and the
generated grammar disagree, the dialect *appears* to support a feature
the parser actually rejects -- a silent failure that's only caught at
benchmark time.

This module renders a small set of self-contained, fully-formed SQL
snippets that mirror the patterns the card advertises, then parses each
against the final grammar. Items that fail to parse become
``card_warnings`` recorded in ``metadata.json``; under
``require_battery_pass`` they abort the package write.

The snippets are constructed from the ``DialectSpec`` directly (no
``DialectEngine`` needed) so the gate runs before any package files
are written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lark import Lark
from lark.exceptions import LarkError

from manysql.spec.dialect import (
    CastSyntax,
    DialectSpec,
    LimitSyntax,
    StringQuote,
    SurfaceSpec,
    WildcardChar,
)


_WILDCARD_LITERAL: dict[WildcardChar, str] = {
    WildcardChar.STAR: "*",
    WildcardChar.DOT: ".",
    WildcardChar.AT: "@",
}


@dataclass(frozen=True)
class CardExample:
    """A self-contained SQL fragment derived from a card pattern."""

    label: str
    source: str


@dataclass(frozen=True)
class CardWarning:
    """A card pattern the dialect grammar rejects."""

    label: str
    source: str
    error: str


@dataclass(frozen=True)
class CardConformanceReport:
    examples: list[CardExample]
    warnings: list[CardWarning]

    @property
    def ok(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        if self.ok:
            return f"card conformance: {len(self.examples)} / {len(self.examples)} OK"
        return (
            f"card conformance: {len(self.examples) - len(self.warnings)} / "
            f"{len(self.examples)} OK; failing: "
            + ", ".join(w.label for w in self.warnings)
        )


def build_card_examples(spec: DialectSpec) -> list[CardExample]:
    """Materialize a fixed set of card-style SQL fragments for ``spec``.

    Each fragment is a complete, self-contained statement that uses the
    dialect's surface forms exactly as the card advertises them. Pure
    function; depends on ``spec.surface`` only.
    """
    s = spec.surface
    examples: list[CardExample] = []
    term = s.statement_terminator if s.requires_semicolon else ""

    sel = s.select_keyword
    frm = s.from_keyword
    whr = s.where_keyword
    gby = s.group_by_keyword
    hav = s.having_keyword
    oby = s.order_by_keyword
    nlk = s.nulls_last_keyword
    nfk = s.nulls_first_keyword
    in_kw = s.in_keyword
    bw = s.between_keyword
    isk = s.is_keyword
    nk = s.null_keyword
    nnk = s.not_keyword
    case = s.case_keyword
    when = s.when_keyword
    then = s.then_keyword
    elsk = s.else_keyword
    end = s.end_keyword
    union = s.union_keyword
    wk = s.with_keyword
    ask = s.as_keyword
    jin = s.join_inner_keyword
    jlt = s.join_left_keyword
    wildcard = _WILDCARD_LITERAL[s.wildcard_char]
    eq = s.eq_op
    add = s.add_op
    concat = s.concat_op
    qot_l, qot_r = _string_quote_pair(s)

    examples.append(
        CardExample(
            "card_select_skeleton",
            f"{sel} id {frm} employees{term}",
        )
    )
    examples.append(
        CardExample(
            "card_filter",
            f"{sel} id {frm} employees {whr} dept_id {eq} 10{term}",
        )
    )
    examples.append(
        CardExample(
            "card_wildcard",
            f"{sel} {wildcard} {frm} employees{term}",
        )
    )
    examples.append(
        CardExample(
            "card_group_having",
            f"{sel} dept_id, COUNT({wildcard}) {ask} n {frm} employees "
            f"{gby} dept_id {hav} COUNT({wildcard}) > 1{term}",
        )
    )
    examples.append(
        CardExample(
            "card_order_by_nulls_last",
            f"{sel} id {frm} employees {oby} dept_id DESC {nlk}{term}",
        )
    )
    examples.append(
        CardExample(
            "card_order_by_nulls_first",
            f"{sel} id {frm} employees {oby} dept_id ASC {nfk}{term}",
        )
    )
    examples.append(
        CardExample(
            "card_in_list",
            f"{sel} id {frm} employees {whr} dept_id {in_kw} (10, 20){term}",
        )
    )
    examples.append(
        CardExample(
            "card_between",
            f"{sel} id {frm} employees {whr} salary {bw} 80000 AND 120000{term}",
        )
    )
    examples.append(
        CardExample(
            "card_is_null",
            f"{sel} id {frm} employees {whr} dept_id {isk} {nk}{term}",
        )
    )
    examples.append(
        CardExample(
            "card_is_not_null",
            f"{sel} id {frm} employees {whr} dept_id {isk} {nnk} {nk}{term}",
        )
    )
    examples.append(
        CardExample(
            "card_case",
            f"{sel} id, {case} {when} salary > 100000 {then} {qot_l}high{qot_r} "
            f"{elsk} {qot_l}low{qot_r} {end} {ask} tier {frm} employees{term}",
        )
    )
    examples.append(
        CardExample(
            "card_inner_join",
            f"{sel} e.id, d.name {frm} employees e {jin} departments d "
            f"ON e.dept_id {eq} d.id{term}",
        )
    )
    examples.append(
        CardExample(
            "card_left_join",
            f"{sel} e.id, d.name {frm} employees e {jlt} departments d "
            f"ON e.dept_id {eq} d.id{term}",
        )
    )
    examples.append(
        CardExample(
            "card_cte",
            f"{wk} high {ask} ({sel} id {frm} employees {whr} salary > 100000) "
            f"{sel} {wildcard} {frm} high{term}",
        )
    )
    examples.append(
        CardExample(
            "card_set_op",
            f"{sel} id {frm} employees {union} ALL {sel} id {frm} departments{term}",
        )
    )
    examples.append(
        CardExample(
            "card_concat",
            f"{sel} name {concat} {qot_l}!{qot_r} {ask} shouted {frm} employees{term}",
        )
    )
    examples.append(
        CardExample(
            "card_arith",
            f"{sel} id, salary {add} 1000 {ask} bumped {frm} employees{term}",
        )
    )

    examples.extend(_limit_examples(spec, sel, frm, term))
    examples.append(_cast_example(spec, sel, frm, term))
    if s.null_safe_eq_op:
        examples.append(
            CardExample(
                "card_null_safe_eq",
                f"{sel} id {frm} employees {whr} dept_id {s.null_safe_eq_op} 10{term}",
            )
        )

    return examples


def validate_card_conformance(
    grammar_text: str,
    examples: list[CardExample],
) -> CardConformanceReport:
    """Parse every ``CardExample`` against ``grammar_text``.

    Items that fail are returned as :class:`CardWarning` objects. A
    grammar-build failure surfaces as a per-example warning so the
    operator can see what specifically the LLM produced.
    """
    if not examples:
        return CardConformanceReport(examples=examples, warnings=[])
    try:
        parser = Lark(grammar_text, start="start", parser="earley")
    except LarkError as e:
        return CardConformanceReport(
            examples=examples,
            warnings=[
                CardWarning(
                    label=ex.label,
                    source=ex.source,
                    error=f"grammar build failed: {e}",
                )
                for ex in examples
            ],
        )
    warnings: list[CardWarning] = []
    for ex in examples:
        try:
            parser.parse(ex.source)
        except Exception as e:
            warnings.append(
                CardWarning(
                    label=ex.label,
                    source=ex.source,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return CardConformanceReport(examples=examples, warnings=warnings)


def _string_quote_pair(surface: SurfaceSpec) -> tuple[str, str]:
    """Return the open/close pair for the dialect's string literal quote."""
    if surface.string_quote == StringQuote.SINGLE:
        return "'", "'"
    if surface.string_quote == StringQuote.DOUBLE:
        return '"', '"'
    if surface.string_quote == StringQuote.BACKTICK:
        return "`", "`"
    return "'", "'"


def _limit_examples(
    spec: DialectSpec, sel: str, frm: str, term: str
) -> list[CardExample]:
    """Render the dialect's LIMIT/OFFSET pattern as a parseable snippet."""
    syntax = spec.surface.limit_syntax
    limit_kw = spec.surface.limit_keyword
    if syntax == LimitSyntax.LIMIT_OFFSET:
        return [
            CardExample(
                "card_limit_offset",
                f"{sel} id {frm} employees {limit_kw} 10 OFFSET 0{term}",
            )
        ]
    if syntax == LimitSyntax.OFFSET_FETCH:
        return [
            CardExample(
                "card_offset_fetch",
                f"{sel} id {frm} employees OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY{term}",
            )
        ]
    if syntax == LimitSyntax.TOP_N:
        return [
            CardExample(
                "card_top_n",
                f"{sel} TOP 10 id {frm} employees{term}",
            )
        ]
    return []


def _cast_example(
    spec: DialectSpec, sel: str, frm: str, term: str
) -> CardExample:
    """Render the dialect's CAST shape (function vs ``::`` postfix)."""
    syntax = spec.surface.cast_syntax
    cast_kw = spec.surface.cast_keyword
    if syntax == CastSyntax.DOUBLE_COLON:
        return CardExample(
            "card_cast_double_colon",
            f"{sel} salary::INT {frm} employees{term}",
        )
    return CardExample(
        "card_cast_function",
        f"{sel} {cast_kw}(salary AS INT) {frm} employees{term}",
    )


__all__ = [
    "CardExample",
    "CardWarning",
    "CardConformanceReport",
    "build_card_examples",
    "validate_card_conformance",
]
