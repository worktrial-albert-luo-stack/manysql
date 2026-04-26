"""Tests for the ``manysql-dialect diff`` data layer and CLI.

The data layer (``compute_battery_diff``) is tested directly so we can assert
on the structured diff without parsing rendered output. The CLI is exercised
through ``main`` with a temp ``--dialects-dir`` to confirm exit codes and
that the rendered output mentions the expected dialect-flavored keywords.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from manysql.codegen import write_dialect_package
from manysql.dialects.__main__ import main as dialect_cli_main
from manysql.dialects.diff import (
    BatteryDiff,
    BatteryDiffItem,
    compute_battery_diff,
    render_battery_diff_table,
    render_battery_diff_unified,
)
from manysql.spec.examples import EXAMPLE_SPECS


@pytest.fixture
def written_moderate(tmp_path: Path) -> Path:
    """Write a moderate-keyword-swap dialect into a temp registry root."""
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    result = write_dialect_package(spec, tmp_path)
    return result.path


def test_compute_battery_diff_pulls_from_battery_json(written_moderate: Path) -> None:
    diff = compute_battery_diff(written_moderate)
    assert isinstance(diff, BatteryDiff)
    assert diff.dialect_name == "moderate_keyword_swap"
    assert diff.source == "battery.json"
    assert diff.total_count >= 20
    # Moderate spec swaps SELECT->PICK and WHERE->COND, so most items must
    # actually differ from the reference.
    assert diff.changed_count >= 15

    scan_all = next(it for it in diff.items if it.label == "scan_all")
    assert scan_all.reference_sql == "SELECT * FROM employees"
    assert scan_all.dialect_sql == "PICK * FROM employees"
    assert scan_all.changed is True


def test_compute_battery_diff_falls_back_to_spec_json(
    written_moderate: Path,
) -> None:
    """If battery.json is missing (older package), recompute from spec.json."""
    (written_moderate / "battery.json").unlink()
    diff = compute_battery_diff(written_moderate)
    assert diff.source == "spec.json (recomputed)"
    assert diff.changed_count >= 15
    scan_all = next(it for it in diff.items if it.label == "scan_all")
    assert scan_all.dialect_sql == "PICK * FROM employees"


def test_compute_battery_diff_raises_when_no_artifacts(tmp_path: Path) -> None:
    empty = tmp_path / "ghost"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        compute_battery_diff(empty)


def test_render_table_includes_dialect_keywords(written_moderate: Path) -> None:
    diff = compute_battery_diff(written_moderate)
    table = render_battery_diff_table(diff)
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=200).print(table)
    out = buf.getvalue()
    assert "PICK" in out
    assert "COND" in out
    assert "SELECT" in out  # reference column intact


def test_render_unified_changed_only_skips_identical_items() -> None:
    """Hand-build a mixed diff so we don't depend on a spec where every
    canonical query happens to be reskinned."""
    diff = BatteryDiff(
        dialect_name="mock",
        items=[
            BatteryDiffItem(
                label="changed",
                reference_sql="SELECT * FROM t",
                dialect_sql="PICK * FROM t",
            ),
            BatteryDiffItem(
                label="identical",
                reference_sql="SELECT 1",
                dialect_sql="SELECT 1",
            ),
        ],
        source="test",
    )
    full = render_battery_diff_unified(diff, only_changed=False)
    changed_only = render_battery_diff_unified(diff, only_changed=True)
    assert "identical" in full
    assert "identical" not in changed_only
    assert "changed" in changed_only
    assert "items reskinned" in changed_only


def test_cli_diff_exit_code_zero(written_moderate: Path) -> None:
    rc = dialect_cli_main(
        [
            "diff",
            "moderate_keyword_swap",
            "--dialects-dir",
            str(written_moderate.parent),
        ]
    )
    assert rc == 0


def test_cli_diff_unknown_dialect_exits_one(tmp_path: Path) -> None:
    rc = dialect_cli_main(
        ["diff", "ghost", "--dialects-dir", str(tmp_path)]
    )
    assert rc == 1


def test_cli_diff_unified_output_mentions_changed_keywords(
    written_moderate: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = dialect_cli_main(
        [
            "diff",
            "moderate_keyword_swap",
            "--dialects-dir",
            str(written_moderate.parent),
            "--unified",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "PICK" in captured.out
    assert "SELECT" in captured.out
