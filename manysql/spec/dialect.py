"""DialectSpec: the input contract to the codegen pipeline.

A DialectSpec describes how a target dialect diverges from the reference. The
codegen agents consume it to emit grammar.lark, lowering.py, semantics.json,
and (optionally) operator overrides.

Design goals:
- Two halves: *surface* (lexical / syntactic) and *semantics* (runtime).
- Every field has a sensible default that matches the reference dialect, so
  a "do nothing" spec produces a copy of the reference.
- Each surface knob is intentionally narrow so an LLM can fill it without
  hallucinating beyond the IR's reach.
- The full set of surface variations is constrained so that codegen can be
  validated mechanically against parse/lowering batteries.

A DialectSpec is *not* the same as a SemanticConfig:
- SemanticConfig is consumed at runtime by the executor and oracle harness.
- DialectSpec is consumed at compile time by codegen agents to write the
  dialect's source files (one of which is a SemanticConfig).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from manysql.spec.semantics import (
    BoolTruthiness,
    CaseFold,
    CountDistinctNull,
    DivByZero,
    IntDivision,
    NullOrder,
    SemanticConfig,
    SetOpDefault,
    StringConcatOp,
    WindowFrameDefault,
)


class DivergenceLevel(str, Enum):
    """Coarse band describing how far from the reference this dialect strays.

    Used to gate test expectations and to bias codegen toward conservative
    rewrites for mild specs and bolder ones for aggressive specs.
    """

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class CommentStyle(str, Enum):
    LINE_DASH_DASH = "line_dash_dash"  # -- comment
    LINE_HASH = "line_hash"            # # comment (MySQL-ish)
    BLOCK_C = "block_c"                # /* ... */
    LINE_DOUBLE_SLASH = "line_double_slash"  # // comment


class StringQuote(str, Enum):
    SINGLE = "single"        # 'text'  (ANSI / Postgres)
    DOUBLE = "double"        # "text"  (some MS / config-dependent)
    BACKTICK = "backtick"    # `text`  (MySQL string for some configs)


class IdentifierQuote(str, Enum):
    DOUBLE = "double"      # ANSI: "ident"
    BACKTICK = "backtick"  # MySQL: `ident`
    BRACKET = "bracket"    # SQL Server: [ident]


class JoinSyntax(str, Enum):
    """Surface form variations for joins. The IR is unaffected."""

    ANSI = "ansi"                          # JOIN ... ON / USING
    KEYWORD_FILTER = "keyword_filter"      # JOIN ... FILTER (...)
    INFIX_TILDE = "infix_tilde"            # left ~JOIN right ON ...
    PIPELINED = "pipelined"                # |> join right on ...


class OrderByPosition(str, Enum):
    """Where ORDER BY appears in the surface query."""

    AFTER_SELECT = "after_select"      # ANSI default
    BEFORE_SELECT = "before_select"    # research/teaching dialects
    INSIDE_FROM_BRACE = "inside_from_brace"


class LimitSyntax(str, Enum):
    LIMIT_OFFSET = "limit_offset"          # LIMIT n OFFSET m (ANSI)
    OFFSET_FETCH = "offset_fetch"          # OFFSET m ROWS FETCH NEXT n ROWS ONLY
    TOP_N = "top_n"                        # SELECT TOP n ...
    SAMPLE_N = "sample_n"                  # SAMPLE n ...
    HEAD_N = "head_n"                      # | head n


class CastSyntax(str, Enum):
    CAST_AS = "cast_as"                # CAST(x AS T)
    DOUBLE_COLON = "double_colon"      # x::T
    CONVERT_FN = "convert_fn"          # CONVERT(T, x)
    AS_FN = "as_fn"                    # AS_TYPE(x, T) — invented form for testing


class CaseSyntax(str, Enum):
    CASE_WHEN = "case_when"        # CASE WHEN ... THEN ... END
    SWITCH = "switch"              # SWITCH(expr, [val1, res1, ...], default)
    PIPELINE = "pipeline"          # x |> ? when=... then=... else=...


class NullLiteral(str, Enum):
    NULL = "NULL"
    NIL = "NIL"
    NONE = "NONE"
    NOTHING = "NOTHING"


class WildcardChar(str, Enum):
    STAR = "star"            # SELECT *  (ANSI)
    DOT = "dot"              # SELECT . (research)
    AT = "at"                # SELECT @


class SetOpPrecedence(str, Enum):
    """Surface-level grouping precedence between UNION / INTERSECT / EXCEPT.

    The IR is unaffected (set-op nodes carry their own kind); this knob only
    decides how a *flat* sequence of set-op branches in the surface SQL
    associates when no parentheses are present.

    - ``ANSI`` (default): all three set ops have the same precedence and
      associate left-to-right, exactly as the reference grammar emits.
    - ``EXCEPT_INTERSECT_TIGHTER``: ``INTERSECT`` and ``EXCEPT`` bind more
      tightly than ``UNION``, so ``A UNION B INTERSECT C`` parses as
      ``A UNION (B INTERSECT C)``. This matches Postgres / DB2 / SQL Server
      and is a common real-world divergence.
    """

    ANSI = "ansi"
    EXCEPT_INTERSECT_TIGHTER = "except_intersect_tighter"


class SurfaceSpec(BaseModel):
    """Lexical and syntactic divergences from the reference.

    Every field defaults to the reference dialect's choice. A spec with all
    defaults will (after codegen) parse the same surface as the reference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ---- Lexical ----
    string_quote: StringQuote = StringQuote.SINGLE
    identifier_quote: IdentifierQuote = IdentifierQuote.DOUBLE
    comment_styles: list[CommentStyle] = Field(
        default_factory=lambda: [CommentStyle.LINE_DASH_DASH, CommentStyle.BLOCK_C]
    )
    null_literal: NullLiteral = NullLiteral.NULL
    wildcard_char: WildcardChar = WildcardChar.STAR

    # ---- Surface keywords (renames) ----
    select_keyword: str = "SELECT"
    from_keyword: str = "FROM"
    where_keyword: str = "WHERE"
    group_by_keyword: str = "GROUP BY"
    having_keyword: str = "HAVING"
    order_by_keyword: str = "ORDER BY"
    limit_keyword: str = "LIMIT"
    distinct_keyword: str = "DISTINCT"
    union_keyword: str = "UNION"
    intersect_keyword: str = "INTERSECT"
    except_keyword: str = "EXCEPT"
    with_keyword: str = "WITH"
    as_keyword: str = "AS"
    join_inner_keyword: str = "JOIN"
    join_left_keyword: str = "LEFT JOIN"
    join_right_keyword: str = "RIGHT JOIN"
    join_full_keyword: str = "FULL JOIN"
    join_cross_keyword: str = "CROSS JOIN"
    case_keyword: str = "CASE"
    when_keyword: str = "WHEN"
    then_keyword: str = "THEN"
    else_keyword: str = "ELSE"
    end_keyword: str = "END"
    cast_keyword: str = "CAST"
    is_keyword: str = "IS"
    not_keyword: str = "NOT"
    null_keyword: str = "NULL"
    in_keyword: str = "IN"
    between_keyword: str = "BETWEEN"
    like_keyword: str = "LIKE"
    ilike_keyword: str = "ILIKE"
    exists_keyword: str = "EXISTS"
    nulls_first_keyword: str = "NULLS FIRST"
    nulls_last_keyword: str = "NULLS LAST"

    # ---- Operators ----
    eq_op: str = "="
    neq_op: list[str] = Field(default_factory=lambda: ["<>", "!="])
    lt_op: str = "<"
    lte_op: str = "<="
    gt_op: str = ">"
    gte_op: str = ">="
    add_op: str = "+"
    sub_op: str = "-"
    mul_op: str = "*"
    div_op: str = "/"
    mod_op: str = "%"
    concat_op: str = "||"
    null_safe_eq_op: Optional[str] = "IS NOT DISTINCT FROM"

    # ---- Higher-level structural choices ----
    join_syntax: JoinSyntax = JoinSyntax.ANSI
    order_by_position: OrderByPosition = OrderByPosition.AFTER_SELECT
    limit_syntax: LimitSyntax = LimitSyntax.LIMIT_OFFSET
    cast_syntax: CastSyntax = CastSyntax.CAST_AS
    case_syntax: CaseSyntax = CaseSyntax.CASE_WHEN
    set_op_precedence: SetOpPrecedence = SetOpPrecedence.ANSI

    # ---- Function aliases ----
    function_aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map canonical name (UPPER) -> dialect surface names. "
        "First entry is the primary; others are accepted aliases.",
    )

    # ---- Misc ----
    requires_semicolon: bool = False
    statement_terminator: Optional[str] = ";"


class SemanticDivergences(BaseModel):
    """A subset view of SemanticConfig for "what differs from reference."

    The codegen pipeline copies these into a full SemanticConfig at emit time;
    fields the spec leaves unset stay at the reference defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identifier_case_fold: Optional[CaseFold] = None
    quoted_identifiers_case_sensitive: Optional[bool] = None
    null_order_default_asc: Optional[NullOrder] = None
    null_order_default_desc: Optional[NullOrder] = None
    null_safe_eq_supported: Optional[bool] = None
    division_by_zero: Optional[DivByZero] = None
    integer_division: Optional[IntDivision] = None
    like_case_sensitive: Optional[bool] = None
    ilike_supported: Optional[bool] = None
    string_concat_op: Optional[StringConcatOp] = None
    set_op_default: Optional[SetOpDefault] = None
    boolean_truthiness: Optional[BoolTruthiness] = None
    count_distinct_null: Optional[CountDistinctNull] = None
    sum_of_empty_returns_null: Optional[bool] = None
    window_default_frame: Optional[WindowFrameDefault] = None
    array_base_index: Optional[int] = None
    group_by_accepts_select_aliases: Optional[bool] = None
    select_resolves_through_group_by: Optional[bool] = None

    def to_semantic_config(self) -> SemanticConfig:
        """Apply this spec's overrides on top of the reference defaults."""
        ref = SemanticConfig.reference().model_dump()
        overrides = {k: v for k, v in self.model_dump().items() if v is not None}
        return SemanticConfig(**{**ref, **overrides})


class DialectSpec(BaseModel):
    """Top-level codegen input.

    Bundles together identity, divergence level, surface form, and semantic
    overrides.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Dialect identifier; becomes the package name.")
    description: str = Field(default="", description="One-line dialect summary.")
    divergence: DivergenceLevel = DivergenceLevel.MILD
    inspired_by: list[str] = Field(
        default_factory=list,
        description="Real-world dialects this one borrows ideas from. "
        "Free-form list; only used in metadata.",
    )

    surface: SurfaceSpec = Field(default_factory=SurfaceSpec)
    semantics: SemanticDivergences = Field(default_factory=SemanticDivergences)

    notes: Optional[str] = None


__all__ = [
    "CaseSyntax",
    "CastSyntax",
    "CommentStyle",
    "DialectSpec",
    "DivergenceLevel",
    "IdentifierQuote",
    "JoinSyntax",
    "LimitSyntax",
    "NullLiteral",
    "OrderByPosition",
    "SemanticDivergences",
    "SetOpPrecedence",
    "StringQuote",
    "SurfaceSpec",
    "WildcardChar",
]
