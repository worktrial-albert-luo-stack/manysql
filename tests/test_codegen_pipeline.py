"""End-to-end codegen tests.

These tests exercise the deterministic codegen path: take an example spec,
emit a dialect package into a temp dir, register it, then run a sample of the
golden query corpus through it and verify against the multi-oracle harness.

The deterministic emitters keep grammar rule names identical to the
reference, so the lowering is reused verbatim and IR output matches exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from lark import Lark

from manysql.codegen import (
    build_package_bundle,
    emit_grammar,
    emit_lowering,
    emit_semantic_config,
    write_dialect_package,
)
from manysql.dialects import DialectRegistry
from manysql.executor import execute
from manysql.golden import GOLDEN_QUERIES
from manysql.oracle import OracleHarness, Verdict
from manysql.spec.examples import EXAMPLE_SPECS
from manysql.storage import CATALOG, schema_of, seed_datasets


def test_bundle_for_default_spec_matches_reference_shape() -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    bundle, grammar_result, lowering_result = build_package_bundle(spec)
    assert grammar_result.ok, grammar_result.report.summary()
    assert lowering_result.ok, lowering_result.report.summary()
    assert "start: statement" in bundle.grammar
    assert "lowering" in bundle.lowering_py.lower()
    assert json.loads(bundle.semantics_json)
    assert json.loads(bundle.metadata_json)["lifecycle"] == "draft"
    spec_payload = json.loads(bundle.spec_json)
    assert spec_payload["name"] == "mild_postgres_ish"
    assert spec_payload["divergence_level"] == "mild"
    battery = json.loads(bundle.battery_json)
    assert battery["parse"]["validation"]["ok"] is True
    assert battery["ir_equivalence"]["validation"]["ok"] is True
    assert "-- scan_all" in bundle.examples_sql


def test_grammar_emitter_renames_keywords() -> None:
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    grammar = emit_grammar(spec)
    assert '"PICK"i' in grammar
    assert '"COND"i' in grammar
    assert '"SORT"i "BY"i' in grammar
    assert '"TAKE"i' in grammar


def test_grammar_emitter_disables_ilike_when_unsupported() -> None:
    spec = EXAMPLE_SPECS["aggressive_alien"]
    grammar = emit_grammar(spec)
    assert "ilike_op" not in grammar


def test_lowering_emitter_rejects_structural_changes() -> None:
    from manysql.spec.dialect import (
        DialectSpec,
        JoinSyntax,
        SurfaceSpec,
    )

    spec = DialectSpec(
        name="experimental",
        surface=SurfaceSpec(join_syntax=JoinSyntax.PIPELINED),
    )
    with pytest.raises(NotImplementedError):
        emit_lowering(spec)


def test_semantic_config_picks_up_overrides() -> None:
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    cfg = json.loads(emit_semantic_config(spec))
    assert cfg["set_op_default"] == "all"
    assert cfg["division_by_zero"] == "null"
    assert cfg["sum_of_empty_returns_null"] is False


def test_write_dialect_package_round_trip(tmp_path: Path) -> None:
    spec = EXAMPLE_SPECS["mild_postgres_ish"]
    result = write_dialect_package(spec, tmp_path)
    assert result.path == tmp_path / spec.name
    for f in [
        "grammar.lark",
        "lowering.py",
        "overrides.py",
        "semantics.json",
        "metadata.json",
        "spec.json",
        "battery.json",
        "examples.sql",
        "__init__.py",
    ]:
        assert (result.path / f).is_file()


def test_battery_artifacts_have_expected_shape(tmp_path: Path) -> None:
    """`battery.json` and `examples.sql` capture the reskinned battery so it
    travels with the dialect package. We check the dialect-flavored surface
    actually shows up (proving the rewriter ran), not just file existence."""
    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    result = write_dialect_package(spec, tmp_path)

    battery = json.loads((result.path / "battery.json").read_text())
    assert "parse" in battery and "ir_equivalence" in battery

    parse_items = battery["parse"]["items"]
    assert len(parse_items) >= 20  # the canonical reference corpus
    assert {"scan_all", "filter_eq", "agg_group_by"} <= {
        item["label"] for item in parse_items
    }
    parse_validation = battery["parse"]["validation"]
    assert parse_validation["ok"] is True
    assert parse_validation["failures"] == []

    ir_items = battery["ir_equivalence"]["items"]
    assert len(ir_items) == len(parse_items)
    # Reference SQL is preserved; dialect SQL is the rewritten form.
    scan_all = next(item for item in ir_items if item["label"] == "scan_all")
    assert scan_all["ref_sql"].upper().startswith("SELECT")
    assert scan_all["dialect_sql"].upper().startswith("PICK")
    assert battery["ir_equivalence"]["validation"]["ok"] is True

    examples = (result.path / "examples.sql").read_text()
    assert f"dialect: {spec.name}" in examples
    assert "-- scan_all" in examples
    # The moderate spec swaps SELECT->PICK, WHERE->COND, LIMIT->TAKE, and
    # GROUP BY->CLUSTER BY; all four should appear in the rewritten queries.
    assert "PICK" in examples
    assert "COND" in examples
    assert "TAKE" in examples
    assert "CLUSTER BY" in examples


@pytest.fixture(scope="module")
def emitted_dialect_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize mild_postgres_ish into a temp registry root and return it."""
    root = tmp_path_factory.mktemp("dialects")
    write_dialect_package(EXAMPLE_SPECS["mild_postgres_ish"], root)
    return root


def test_emitted_dialect_loads_via_registry(emitted_dialect_root: Path) -> None:
    sys.path.insert(0, str(emitted_dialect_root.parent))
    try:
        registry = DialectRegistry(root=emitted_dialect_root)
        engine = registry.load("mild_postgres_ish")
        assert engine.semantics.integer_division.value == "truncate"
        assert engine.semantics.division_by_zero.value == "error"
    finally:
        if str(emitted_dialect_root.parent) in sys.path:
            sys.path.remove(str(emitted_dialect_root.parent))


@pytest.fixture(scope="module")
def emitted_engine(emitted_dialect_root: Path):
    registry = DialectRegistry(root=emitted_dialect_root)
    return registry.load("mild_postgres_ish")


@pytest.fixture(scope="module")
def emitted_parser(emitted_engine):
    return Lark(emitted_engine.grammar_text, start="start", parser="earley")


@pytest.fixture(scope="module")
def catalog():
    return seed_datasets()


@pytest.fixture(scope="module")
def schemas():
    return {name: schema_of(name) for name in CATALOG}


@pytest.fixture(scope="module")
def harness():
    return OracleHarness()


# A representative slice of the golden corpus. Everything cross_dialect should
# work for the mild spec; we sample to keep test wall-time low.
SAMPLE_IDS = [
    "scan_employees",
    "filter_eq",
    "filter_in_list",
    "filter_like",
    "project_arithmetic",
    "project_case",
    "join_inner_on",
    "join_left_on",
    "agg_sum_avg_min_max",
    "agg_group_by",
    "sort_asc",
    "sort_limit",
    "distinct_one_col",
    "union_distinct",
    "subq_in_uncorrelated",
    "cte_simple",
    "window_row_number_partitioned",
]


@pytest.mark.parametrize("qid", SAMPLE_IDS)
def test_emitted_dialect_executes_golden_sample(
    qid: str,
    emitted_engine,
    emitted_parser: Lark,
    catalog,
    schemas,
    harness: OracleHarness,
) -> None:
    q = next(q for q in GOLDEN_QUERIES if q.id == qid)
    tree = emitted_parser.parse(q.sql)
    plan = emitted_engine.lowering.lower(tree, emitted_engine.semantics, schemas)
    actual = execute(plan, emitted_engine.semantics, catalog)
    report = harness.verify(plan, emitted_engine.semantics, catalog, actual)
    assert report.verdict == Verdict.PASS, (
        f"[{qid}] verdict={report.verdict} primary={report.primary} "
        f"reason={report.actual_vs_primary_reason}"
    )


@pytest.fixture(scope="module")
def emitted_moderate_dialect_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dialects_moderate")
    write_dialect_package(EXAMPLE_SPECS["moderate_keyword_swap"], root)
    return root


@pytest.fixture(scope="module")
def emitted_moderate_engine(emitted_moderate_dialect_root: Path):
    registry = DialectRegistry(root=emitted_moderate_dialect_root)
    return registry.load("moderate_keyword_swap")


@pytest.fixture(scope="module")
def emitted_moderate_parser(emitted_moderate_engine):
    return Lark(
        emitted_moderate_engine.grammar_text, start="start", parser="earley"
    )


@pytest.fixture(scope="module")
def emitted_aggressive_dialect_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("dialects_aggressive")
    write_dialect_package(EXAMPLE_SPECS["aggressive_alien"], root)
    return root


@pytest.fixture(scope="module")
def emitted_aggressive_engine(emitted_aggressive_dialect_root: Path):
    registry = DialectRegistry(root=emitted_aggressive_dialect_root)
    return registry.load("aggressive_alien")


@pytest.fixture(scope="module")
def emitted_aggressive_parser(emitted_aggressive_engine):
    return Lark(
        emitted_aggressive_engine.grammar_text, start="start", parser="earley"
    )


def test_aggressive_dialect_passes_batteries() -> None:
    """The deterministic codegen path covers the aggressive spec when its
    surface knobs (LimitSyntax.OFFSET_FETCH, NULL literal renames, etc.)
    are within the templated rewrites."""
    from manysql.codegen.grammar_agent import generate_grammar
    from manysql.codegen.lowering_agent import generate_lowering

    spec = EXAMPLE_SPECS["aggressive_alien"]
    g = generate_grammar(spec)
    assert g.ok, g.report.summary()
    l = generate_lowering(spec, grammar_text=g.grammar)
    assert l.ok, l.report.summary()


def test_aggressive_dialect_executes_surface_rewritten_battery(
    emitted_aggressive_engine,
    emitted_aggressive_parser: Lark,
    catalog,
    schemas,
    harness: OracleHarness,
) -> None:
    """Apply the aggressive spec's surface to a slice of the golden corpus
    and confirm the dialect parses+lowers+executes them, matching the
    multi-oracle harness."""
    from manysql.codegen.parse_battery import apply_surface

    spec = EXAMPLE_SPECS["aggressive_alien"]
    sample = [
        "scan_employees",
        "filter_eq",
        "filter_in_list",
        "project_arithmetic",
        "join_inner_on",
        "agg_group_by",
        "sort_asc",
        "sort_limit",
        "distinct_one_col",
    ]
    failed: list[str] = []
    for qid in sample:
        q = next(q for q in GOLDEN_QUERIES if q.id == qid)
        rewritten = apply_surface(q.sql, spec.surface)
        try:
            tree = emitted_aggressive_parser.parse(rewritten)
            plan = emitted_aggressive_engine.lowering.lower(
                tree, emitted_aggressive_engine.semantics, schemas
            )
            actual = execute(plan, emitted_aggressive_engine.semantics, catalog)
            report = harness.verify(
                plan, emitted_aggressive_engine.semantics, catalog, actual
            )
        except Exception as exc:
            failed.append(f"{qid}: {type(exc).__name__}: {exc}")
            continue
        if report.verdict == Verdict.FAIL:
            failed.append(
                f"{qid}: verdict={report.verdict.value} "
                f"reason={report.actual_vs_primary_reason}"
            )
    assert not failed, (
        f"aggressive dialect failed {len(failed)} queries: "
        + "; ".join(failed[:5])
    )


def test_moderate_dialect_executes_surface_rewritten_battery(
    emitted_moderate_engine,
    emitted_moderate_parser: Lark,
    catalog,
    schemas,
    harness: OracleHarness,
) -> None:
    """Apply the moderate spec's surface rewriter to a slice of the golden
    corpus and confirm the emitted dialect parses+lowers+executes them and
    each result agrees with the multi-oracle harness."""
    from manysql.codegen.parse_battery import apply_surface

    spec = EXAMPLE_SPECS["moderate_keyword_swap"]
    sample = [
        "scan_employees",
        "filter_eq",
        "filter_in_list",
        "project_arithmetic",
        "join_inner_on",
        "agg_group_by",
        "sort_asc",
        "sort_limit",
        "distinct_one_col",
    ]
    failed: list[str] = []
    for qid in sample:
        q = next(q for q in GOLDEN_QUERIES if q.id == qid)
        rewritten = apply_surface(q.sql, spec.surface)
        try:
            tree = emitted_moderate_parser.parse(rewritten)
            plan = emitted_moderate_engine.lowering.lower(
                tree, emitted_moderate_engine.semantics, schemas
            )
            actual = execute(plan, emitted_moderate_engine.semantics, catalog)
            report = harness.verify(
                plan, emitted_moderate_engine.semantics, catalog, actual
            )
        except Exception as exc:
            failed.append(f"{qid}: {type(exc).__name__}: {exc}")
            continue
        if report.verdict == Verdict.FAIL:
            failed.append(
                f"{qid}: verdict={report.verdict.value} "
                f"reason={report.actual_vs_primary_reason}"
            )
    assert not failed, (
        f"moderate dialect failed {len(failed)} queries: "
        + "; ".join(failed[:5])
    )


def test_mild_dialect_full_cross_dialect_corpus(
    emitted_engine,
    emitted_parser: Lark,
    catalog,
    schemas,
    harness: OracleHarness,
) -> None:
    """End-to-end: every cross_dialect golden query passes against the
    freshly emitted mild_postgres_ish dialect.

    This is the canonical "first synthetic dialect end-to-end" check.
    """
    cross = [q for q in GOLDEN_QUERIES if q.cross_dialect]
    failed: list[str] = []
    for q in cross:
        try:
            tree = emitted_parser.parse(q.sql)
            plan = emitted_engine.lowering.lower(
                tree, emitted_engine.semantics, schemas
            )
            actual = execute(plan, emitted_engine.semantics, catalog)
            report = harness.verify(
                plan, emitted_engine.semantics, catalog, actual
            )
        except Exception as exc:
            failed.append(f"{q.id}: {type(exc).__name__}: {exc}")
            continue
        if report.verdict == Verdict.FAIL:
            failed.append(
                f"{q.id}: verdict={report.verdict.value} "
                f"reason={report.actual_vs_primary_reason}"
            )
    # NEEDS_REVIEW (oracles disagree among themselves) is acceptable — it's
    # a known dataset-level ambiguity, not a defect in our executor.
    assert not failed, (
        f"{len(failed)}/{len(cross)} cross-dialect queries failed against "
        f"the emitted mild_postgres_ish dialect:\n  - "
        + "\n  - ".join(failed[:10])
        + ("\n  ..." if len(failed) > 10 else "")
    )
