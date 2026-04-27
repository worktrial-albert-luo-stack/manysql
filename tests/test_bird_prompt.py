"""Surface tests for BIRD prompt rendering.

These tests don't require the BIRD HuggingFace dataset or any of the
~5GB SQLite database files; they construct a synthetic
:class:`BirdEntry` directly and exercise the prompt formatter +
schema-prompt contract.
"""

from __future__ import annotations

from train.env.bird import (
    _BIRD_SCHEMA_PROMPT,
    BirdEntry,
    BirdTableInfo,
    _format_type_label,
    _render_user_prompt,
)


def test_format_type_label_passes_through_non_text() -> None:
    assert _format_type_label("INT", "INTEGER") == "INT"
    assert _format_type_label("FLOAT", "REAL") == "FLOAT"
    assert _format_type_label("BOOL", "BOOLEAN") == "BOOL"


def test_format_type_label_marks_text_dates() -> None:
    # Common BIRD declarations.
    assert _format_type_label("TEXT", "DATETIME") == "TEXT, was DATETIME"
    assert _format_type_label("TEXT", "DATE") == "TEXT, was DATE"
    assert _format_type_label("TEXT", "TIMESTAMP") == "TEXT, was TIMESTAMP"
    # Parenthesized size hints get trimmed.
    assert (
        _format_type_label("TEXT", "DATETIME(3)") == "TEXT, was DATETIME"
    )
    # Plain TEXT/VARCHAR doesn't get the annotation.
    assert _format_type_label("TEXT", "VARCHAR(255)") == "TEXT"
    assert _format_type_label("TEXT", "") == "TEXT"


def test_schema_prompt_mentions_text_date_contract() -> None:
    """The system-prompt schema slot must spell out the CAST contract.

    This is the only place the model learns that TEXT columns can
    actually hold date strings, so any rewrite that drops this
    guidance should break this test loudly.
    """
    p = _BIRD_SCHEMA_PROMPT
    assert "DATE/TIME COLUMNS" in p
    assert "ISO" in p
    assert "CAST" in p
    assert "EXTRACT" in p


def test_render_user_prompt_includes_was_datetime_annotation() -> None:
    entry = BirdEntry(
        question_id=1,
        db_id="sample_db",
        question="how many sales happened in 2024?",
        evidence="",
        sql="SELECT COUNT(*) FROM sales WHERE strftime('%Y', sold_on) = '2024'",
        difficulty="simple",
        db_path="/tmp/ignored.sqlite",
        tables=[
            BirdTableInfo(
                original_name="sales",
                catalog_table_name="sample_db__sales",
                safe_columns=["c_id", "c_amount", "c_sold_on", "c_note"],
                original_columns=["id", "amount", "sold_on", "note"],
                types=["INT", "FLOAT", "TEXT", "TEXT"],
                sqlite_types=["INTEGER", "REAL", "DATETIME", "TEXT"],
                sample_rows=[
                    {
                        "c_id": 1,
                        "c_amount": 10.0,
                        "c_sold_on": "2024-01-15",
                        "c_note": "first",
                    },
                ],
                n_rows=1,
            )
        ],
    )
    out = _render_user_prompt(entry)
    # Date-like column shows the original SQLite affinity.
    assert "c_sold_on  <-  sold_on  (TEXT, was DATETIME)" in out
    # Plain TEXT column stays plain.
    assert "c_note  <-  note  (TEXT)" in out
    # Numeric columns unchanged.
    assert "c_amount  <-  amount  (FLOAT)" in out
    assert "c_id  <-  id  (INT)" in out


def test_render_user_prompt_tolerates_legacy_entries_without_sqlite_types() -> None:
    """A BirdTableInfo built before this field existed must still render."""
    entry = BirdEntry(
        question_id=2,
        db_id="legacy_db",
        question="x",
        evidence="",
        sql="SELECT 1",
        difficulty="simple",
        db_path="/tmp/ignored.sqlite",
        tables=[
            BirdTableInfo(
                original_name="t",
                catalog_table_name="legacy_db__t",
                safe_columns=["c_a"],
                original_columns=["a"],
                types=["INT"],
                # sqlite_types defaults to []
                sample_rows=[],
                n_rows=0,
            )
        ],
    )
    out = _render_user_prompt(entry)
    assert "c_a  <-  a  (INT)" in out
