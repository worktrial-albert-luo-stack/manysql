"""SemanticConfig: runtime knobs honored by the executor.

Every knob here corresponds to a *silent semantic divergence* between real-world
SQL dialects. The executor reads this config at every behavior decision point,
so two dialects with the same IR plan but different SemanticConfigs really do
produce different output.

Adding a knob is *additive*: existing dialects stay correct because every knob
has a default value taken from the reference dialect (a near-ANSI / DuckDB-aligned
baseline).

See manysql/dialects/_reference/semantics.json for the canonical reference values.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CaseFold(str, Enum):
    LOWER = "lower"      # Postgres
    UPPER = "upper"      # Snowflake / Oracle
    PRESERVE = "preserve"  # SQL Server / case-sensitive identifiers


class NullOrder(str, Enum):
    FIRST = "first"
    LAST = "last"


class DivByZero(str, Enum):
    NULL = "null"     # MySQL default
    ERROR = "error"   # Postgres
    INF = "inf"       # IEEE-754 floats only; integer is still error


class IntDivision(str, Enum):
    TRUNCATE = "truncate"  # 1/2 -> 0 (Postgres int/int, C-style)
    PROMOTE = "promote"    # 1/2 -> 0.5 (MySQL non-strict, SQLite)


class SetOpDefault(str, Enum):
    DISTINCT = "distinct"  # ANSI standard for UNION/INTERSECT/EXCEPT
    ALL = "all"


class BoolTruthiness(str, Enum):
    STRICT = "strict"   # only true/false; non-bool requires explicit cast
    C_STYLE = "c_style"  # 0 -> false, nonzero -> true; '' -> false


class CountDistinctNull(str, Enum):
    EXCLUDED = "excluded"  # ANSI: COUNT(DISTINCT x) ignores NULL (most engines)
    INCLUDED = "included"  # very rare; some research dialects


class WindowFrameDefault(str, Enum):
    """Default frame when a window has ORDER BY but no explicit frame clause."""

    RANGE_UNBOUNDED_TO_CURRENT = "range_unbounded_to_current"  # SQL standard
    ROWS_UNBOUNDED_TO_CURRENT = "rows_unbounded_to_current"
    ROWS_UNBOUNDED_TO_UNBOUNDED = "rows_unbounded_to_unbounded"


class StringConcatOp(str, Enum):
    PIPE_PIPE = "||"            # ANSI / Postgres / Oracle / SQLite
    PLUS = "+"                  # SQL Server
    CONCAT_FN_ONLY = "CONCAT_only"  # MySQL strict


class SetOpPrecedenceMode(str, Enum):
    """Runtime grouping precedence between UNION / INTERSECT / EXCEPT.

    Mirrors ``manysql.spec.dialect.SetOpPrecedence`` but lives here so the
    executor's lowering can consult it directly without importing codegen-
    side enums. ``ANSI`` (default) folds set-op branches left-to-right with
    equal precedence; ``EXCEPT_INTERSECT_TIGHTER`` folds INTERSECT/EXCEPT
    runs first, then UNIONs around them (Postgres / DB2 / SQL Server).
    """

    ANSI = "ansi"
    EXCEPT_INTERSECT_TIGHTER = "except_intersect_tighter"


class CoercionRule(BaseModel):
    """One row of the implicit-coercion matrix.

    Read as: "when an operator/comparison sees `from_type` and expects
    `to_type`, the value is coerced according to `mode`."
    """

    model_config = ConfigDict(frozen=True)

    from_type: str  # IRType.kind value
    to_type: str
    mode: str  # "implicit" | "explicit_only" | "forbidden"


class SemanticConfig(BaseModel):
    """Runtime semantic knobs honored by the executor and oracles.

    Defaults align with the reference dialect (near-ANSI / DuckDB-compatible).
    Each field's docstring names a representative real-world dialect for context.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Identifier and lexical rules ---

    identifier_case_fold: CaseFold = Field(
        default=CaseFold.LOWER,
        description="Postgres lowers, Snowflake upper, SQL Server preserves.",
    )
    quoted_identifiers_case_sensitive: bool = Field(
        default=True,
        description="True everywhere except some MySQL configurations.",
    )

    # --- Null handling and three-valued logic ---

    null_order_default_asc: NullOrder = Field(
        default=NullOrder.LAST,
        description="Default null order for ASC sort. Postgres LAST, MySQL FIRST.",
    )
    null_order_default_desc: NullOrder = Field(
        default=NullOrder.FIRST,
        description="Default null order for DESC sort.",
    )
    null_safe_eq_supported: bool = Field(
        default=True,
        description="Whether IS NOT DISTINCT FROM (or <=>) is supported.",
    )

    # --- Arithmetic ---

    division_by_zero: DivByZero = Field(default=DivByZero.NULL)
    integer_division: IntDivision = Field(default=IntDivision.PROMOTE)

    # --- Coercion ---

    implicit_coercion: list[CoercionRule] = Field(
        default_factory=list,
        description="Pairs that allow implicit cross-type coercion. "
        "Empty list means strict (only same-kind comparisons).",
    )

    # --- String / pattern matching ---

    like_case_sensitive: bool = Field(default=True)
    ilike_supported: bool = Field(default=True)
    string_concat_op: StringConcatOp = Field(default=StringConcatOp.PIPE_PIPE)

    # --- Set operations ---

    set_op_default: SetOpDefault = Field(
        default=SetOpDefault.DISTINCT,
        description="Default for UNION/INTERSECT/EXCEPT when ALL/DISTINCT is omitted.",
    )
    set_op_precedence: SetOpPrecedenceMode = Field(
        default=SetOpPrecedenceMode.ANSI,
        description="ANSI: equal precedence, left-to-right. "
        "EXCEPT_INTERSECT_TIGHTER: Postgres/DB2/SQL Server style — "
        "INTERSECT and EXCEPT bind tighter than UNION.",
    )

    # --- Boolean / aggregation edge cases ---

    boolean_truthiness: BoolTruthiness = Field(default=BoolTruthiness.STRICT)
    count_distinct_null: CountDistinctNull = Field(default=CountDistinctNull.EXCLUDED)
    sum_of_empty_returns_null: bool = Field(
        default=True,
        description="ANSI: yes (NULL). Some dialects return 0.",
    )

    # --- SELECT / GROUP BY scope rules ---

    group_by_accepts_select_aliases: bool = Field(
        default=True,
        description="Whether ``GROUP BY <alias>`` resolves to a SELECT-list "
        "alias when no real column of that name is in scope. "
        "PostgreSQL/SQLite/MySQL/DuckDB accept this; pure ANSI rejects it. "
        "Set False for a strictly-ANSI dialect.",
    )
    select_resolves_through_group_by: bool = Field(
        default=True,
        description="Whether SELECT items that are structurally identical to "
        "a GROUP BY expression are rewritten to reference the GROUP BY "
        "output column (so the projection works post-aggregate). True in "
        "every mainstream SQL engine; False forces the user/LLM to reuse "
        "the canonical column name explicitly in SELECT, which a few "
        "research dialects prefer.",
    )

    # --- Window functions ---

    window_default_frame: WindowFrameDefault = Field(
        default=WindowFrameDefault.RANGE_UNBOUNDED_TO_CURRENT,
        description="Default frame when ORDER BY is present but FRAME is omitted.",
    )

    # --- Array index base (placeholder for Tier B) ---

    array_base_index: int = Field(
        default=1,
        description="Postgres/Snowflake 1, Trino/Spark 0. "
        "Placeholder until Tier-B array extension lands.",
    )

    # --- Surface aliasing (purely lexical, but lives here for one-stop config) ---

    function_aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map from canonical IR function name to "
        "list of dialect-surface names (first is the primary).",
    )
    keyword_aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map from canonical clause keyword (e.g. 'LIMIT') to "
        "list of dialect-surface forms (e.g. ['CAP', 'TOP']).",
    )

    # --- Optional metadata ---

    notes: Optional[str] = Field(
        default=None, description="Free-form notes about non-knob divergences."
    )

    @classmethod
    def reference(cls) -> "SemanticConfig":
        """The reference dialect's defaults, used as the codegen template baseline."""
        return cls()
