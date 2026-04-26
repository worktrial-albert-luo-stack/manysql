"""Codegen pipeline that turns a DialectSpec into a fully-functional dialect.

Architecture:
    DialectSpec ---> emit_semantic_config ---> semantics.json
                ---> emit_metadata        ---> metadata.json
                ---> emit_spec            ---> spec.json
                ---> emit_grammar         ---> grammar.lark
                ---> emit_lowering        ---> lowering.py
                ---> emit_init            ---> __init__.py
    Combined: write_dialect_package(spec, root)

Each emitter returns text; `write_dialect_package` is the only function that
touches the filesystem. This makes everything trivially unit-testable and
keeps the codegen pure.

LLM-driven emitters (grammar, lowering) accept an optional LLMClient. When
omitted, they fall back to deterministic templates seeded from the reference
dialect. This lets us:
- run end-to-end without keys (great for CI),
- iterate the prompts independently of the framework, and
- diff LLM output against the deterministic baseline.
"""

from manysql.codegen.battery_emit import emit_battery_json, emit_examples_sql
from manysql.codegen.config_emit import emit_semantic_config
from manysql.codegen.grammar_agent import (
    GrammarAgentResult,
    GrammarAttempt,
    generate_grammar,
)
from manysql.codegen.grammar_emit import emit_grammar
from manysql.codegen.ir_battery import (
    IRBatteryItem,
    IRDivergence,
    IREquivalenceReport,
    build_ir_battery,
    validate_lowering,
)
from manysql.codegen.lowering_agent import (
    LoweringAgentResult,
    LoweringAttempt,
    generate_lowering,
)
from manysql.codegen.lowering_emit import emit_lowering
from manysql.codegen.metadata_emit import emit_metadata, emit_spec_json
from manysql.codegen.overrides_emit import emit_overrides
from manysql.codegen.overrides_loader import OverrideImportError, load_overrides
from manysql.codegen.parse_battery import (
    BatteryFailure,
    BatteryItem,
    ValidationReport,
    apply_surface,
    build_parse_battery,
    validate_grammar,
)
from manysql.codegen.pipeline import (
    PackageBundle,
    PackageWriteResult,
    build_package_bundle,
    write_dialect_package,
)

__all__ = [
    "BatteryFailure",
    "BatteryItem",
    "GrammarAgentResult",
    "GrammarAttempt",
    "IRBatteryItem",
    "IRDivergence",
    "IREquivalenceReport",
    "LoweringAgentResult",
    "LoweringAttempt",
    "OverrideImportError",
    "PackageBundle",
    "PackageWriteResult",
    "ValidationReport",
    "apply_surface",
    "build_ir_battery",
    "build_package_bundle",
    "build_parse_battery",
    "emit_battery_json",
    "emit_examples_sql",
    "emit_grammar",
    "emit_lowering",
    "emit_metadata",
    "emit_overrides",
    "emit_semantic_config",
    "emit_spec_json",
    "generate_grammar",
    "generate_lowering",
    "load_overrides",
    "validate_grammar",
    "validate_lowering",
    "write_dialect_package",
]
