"""Coverage for DATE_DIFF unit expansion (week/month/quarter/year/sub-day).

Both the Polars executor and the reference interpreter must agree on
every unit. FENCEPOST semantics (count boundaries crossed) is the
canonical behavior for calendar units; sub-day units use total elapsed
time truncated toward zero. Both match sqlite's STRFTIME-arithmetic
convention used in BIRD gold SQL.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from manysql.executor import execute
from manysql.ir import (
    ColumnRef,
    FuncCall,
    Literal,
    Project,
    Scan,
)
from manysql.ir.types import DATE_T, INT, TEXT, TIMESTAMP
from manysql.oracle import ReferenceInterpreter
from manysql.spec import SemanticConfig
from manysql.storage import schema_of, seed_datasets


def _project_diff(unit: str, a: object, b: object, *, t=DATE_T):
    """Build a Project plan computing DATE_DIFF(unit, a, b) on a 1-row table.

    Picks ``regions`` because its row count is small and it has no
    DATE columns of its own, so the constant subexpression cleanly
    produces one result row per input row.
    """
    rgn = Scan(table_name="regions", columns=schema_of("regions"))
    return Project(
        input=rgn,
        projections=(
            (
                "diff",
                FuncCall(
                    name="DATE_DIFF",
                    args=(
                        Literal(unit, TEXT),
                        Literal(a, t),
                        Literal(b, t),
                    ),
                ),
            ),
        ),
        output_types=(INT,),
    )


@pytest.mark.parametrize(
    "unit, a, b, expected",
    [
        # Days
        ("day", date(2024, 1, 1), date(2024, 1, 31), 30),
        ("days", date(2024, 1, 31), date(2024, 1, 1), -30),
        # Weeks (truncated toward zero, NOT floored toward -inf)
        ("week", date(2024, 1, 1), date(2024, 1, 8), 1),
        ("week", date(2024, 1, 1), date(2024, 1, 7), 0),
        ("weeks", date(2024, 1, 8), date(2024, 1, 1), -1),
        # 6-day negative delta: trunc -> 0, floor would be -1.
        ("week", date(2024, 1, 7), date(2024, 1, 1), 0),
        # Months — FENCEPOST: count boundaries crossed
        ("month", date(2024, 1, 31), date(2024, 2, 1), 1),
        ("month", date(2024, 2, 1), date(2024, 1, 31), -1),
        ("months", date(2024, 1, 1), date(2025, 1, 1), 12),
        # Quarters
        ("quarter", date(2024, 1, 1), date(2024, 4, 1), 1),
        ("quarter", date(2024, 1, 1), date(2024, 12, 31), 3),
        # Years
        ("year", date(2024, 6, 15), date(2026, 1, 1), 2),
        ("years", date(2026, 1, 1), date(2024, 6, 15), -2),
    ],
)
def test_date_diff_calendar_units(unit, a, b, expected):
    plan = _project_diff(unit, a, b)
    catalog = seed_datasets()
    semantics = SemanticConfig.reference()

    pl_df = execute(plan, semantics, catalog)
    assert pl_df["diff"].to_list() == [expected] * pl_df.height

    ref = ReferenceInterpreter().evaluate(plan, semantics, catalog)
    assert ref.error is None
    assert ref.rows is not None
    assert ref.rows["diff"].to_list() == [expected] * ref.rows.height


@pytest.mark.parametrize(
    "unit, a, b, expected",
    [
        ("hour", datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 5), 5),
        ("hours", datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 0), -5),
        ("minute", datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 1, 30), 90),
        ("second", datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 30), 30),
    ],
)
def test_date_diff_subday_units(unit, a, b, expected):
    """Sub-day units: total elapsed truncated toward zero."""
    plan = _project_diff(unit, a, b, t=TIMESTAMP)
    catalog = seed_datasets()
    semantics = SemanticConfig.reference()

    pl_df = execute(plan, semantics, catalog)
    assert pl_df["diff"].to_list() == [expected] * pl_df.height

    ref = ReferenceInterpreter().evaluate(plan, semantics, catalog)
    assert ref.error is None
    assert ref.rows is not None
    assert ref.rows["diff"].to_list() == [expected] * ref.rows.height


def test_date_diff_unknown_unit_raises():
    plan = _project_diff("fortnight", date(2024, 1, 1), date(2024, 1, 15))
    with pytest.raises(NotImplementedError, match="DATE_DIFF unit: fortnight"):
        execute(plan, SemanticConfig.reference(), seed_datasets())
