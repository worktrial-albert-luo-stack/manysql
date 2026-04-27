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

from manysql.codegen.batch import (
    CAMPAIGNS_DIRNAME,
    CampaignBrief,
    CampaignConfig,
    CampaignReporter,
    CampaignResult,
    LedgerEntry,
    THEME_CHOICES,
    ThemeLiteral,
    design_dialect_batch,
    expand_campaign_brief,
    run_campaign,
    write_campaign_manifest,
)
from manysql.codegen.battery_emit import emit_battery_json, emit_examples_sql
from manysql.codegen.config_emit import emit_semantic_config
from manysql.codegen.effects_emit import emit_effects
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
from manysql.codegen.overrides_loader import (
    OverrideImportError,
    load_overrides,
    load_sandboxed_module,
)
from manysql.codegen.passes_emit import emit_passes
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
    "CAMPAIGNS_DIRNAME",
    "CampaignBrief",
    "CampaignConfig",
    "CampaignReporter",
    "CampaignResult",
    "GrammarAgentResult",
    "GrammarAttempt",
    "IRBatteryItem",
    "IRDivergence",
    "IREquivalenceReport",
    "LedgerEntry",
    "LoweringAgentResult",
    "LoweringAttempt",
    "OverrideImportError",
    "PackageBundle",
    "PackageWriteResult",
    "THEME_CHOICES",
    "ThemeLiteral",
    "ValidationReport",
    "apply_surface",
    "build_ir_battery",
    "build_package_bundle",
    "build_parse_battery",
    "design_dialect_batch",
    "emit_battery_json",
    "emit_effects",
    "emit_examples_sql",
    "emit_grammar",
    "emit_lowering",
    "emit_metadata",
    "emit_overrides",
    "emit_passes",
    "emit_semantic_config",
    "emit_spec_json",
    "expand_campaign_brief",
    "generate_grammar",
    "generate_lowering",
    "load_overrides",
    "load_sandboxed_module",
    "run_campaign",
    "validate_grammar",
    "validate_lowering",
    "write_campaign_manifest",
    "write_dialect_package",
]
