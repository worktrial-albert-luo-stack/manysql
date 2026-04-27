"""Dialect registry: backend-swappable storage and lifecycle for generated dialects.

A dialect is a bundle:
    spec.json         (DialectSpec - the input that drove generation)
    grammar.lark      (Lark grammar)
    lowering.py       (parse-tree -> IR module; must define `lower(tree, config, catalog) -> Plan`)
    semantics.json    (SemanticConfig)
    overrides.py      (optional: per-function/operator implementations the
                       canonical executor doesn't natively support)
    passes.py         (optional: Plan -> Plan rewrites that run between
                       lowering and execution; for dialects whose surface
                       requires non-canonical IR markers that need to be
                       desugared to canonical IR)
    effects.py        (optional: named handlers swapped into executor
                       decision points; for runtime divergences whose
                       space isn't a small closed enum)
    metadata.json     (model used, prompts, retry log, lifecycle state)
    battery.json      (parse + IR-equivalence battery in dialect surface,
                       plus the latest validation summary)
    examples.sql      (the same battery rendered as human-readable SQL)
    validation.json   (last harness run summary)

v1 backend = on-disk Python packages under manysql/dialects/<name>/.
Tier 2 (SQLite metadata DB + on-disk blobs) lives behind the same interface.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Protocol

from manysql.spec.semantics import SemanticConfig

DIALECTS_DIR = Path(__file__).parent
RESERVED_NAMES = {"_reference", "_registry.db", "registry"}


class Lifecycle(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    GENERATED = "generated"
    VALIDATED = "validated"
    VALIDATED_MANUAL = "validated_manual"  # manual-review oracle tier
    NEEDS_REVIEW = "needs_review"          # inter-oracle disagreement
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class GenerationMetadata:
    model: Optional[str] = None
    provider: Optional[str] = None
    prompts: dict[str, str] = field(default_factory=dict)
    retry_log: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompts": self.prompts,
            "retry_log": self.retry_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationMetadata":
        return cls(
            model=d.get("model"),
            provider=d.get("provider"),
            prompts=d.get("prompts", {}),
            retry_log=d.get("retry_log", []),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ValidationRun:
    """One validation-harness run summary."""

    timestamp: str
    oracle_versions: dict[str, str]  # {oracle_name: version}
    total_plans: int
    passed: int
    failed: int
    needs_review: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class LowererProto(Protocol):
    """Every dialect's lowering.py must expose `lower`."""

    def lower(self, tree: Any, config: SemanticConfig) -> Any:  # returns Plan
        ...


@dataclass
class DialectEngine:
    """A loaded, runnable dialect engine.

    The grammar text is loaded but Lark parser construction is deferred to the
    caller (the executor module owns that, so it can choose lalr/earley/etc).

    `overrides`, `passes`, and `effects` are the three optional, per-dialect
    extension lanes. All three are loaded as Python modules and consulted by
    the runtime when present; absent ones fall back to canonical behavior.
    """

    name: str
    spec: dict[str, Any]
    grammar_text: str
    lowering: ModuleType
    semantics: SemanticConfig
    overrides: Optional[ModuleType] = None
    passes: Optional[ModuleType] = None
    effects: Optional[ModuleType] = None
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
    lifecycle: Lifecycle = Lifecycle.GENERATED


@dataclass
class DialectRecord:
    """Lightweight registry entry without loaded modules. Used by `list()` and `lifecycle()`."""

    name: str
    lifecycle: Lifecycle
    metadata: GenerationMetadata
    last_validation: Optional[ValidationRun] = None


class DialectRegistry:
    """On-disk Tier-1 backend.

    Layout under `root` (default = manysql/dialects/):
        <name>/
            spec.json
            grammar.lark
            lowering.py
            semantics.json
            overrides.py        (optional)
            passes.py           (optional)
            effects.py          (optional)
            metadata.json
            battery.json        (parse + IR battery in dialect surface,
                                 plus the latest validation summary)
            examples.sql        (battery rendered as human-readable SQL)
            validation.json     (optional, written by the harness)
            __init__.py
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or DIALECTS_DIR

    def list(self, *, include_reference: bool = False) -> list[str]:
        out: list[str] = []
        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue
            if p.name in RESERVED_NAMES and not (
                include_reference and p.name == "_reference"
            ):
                continue
            if not (p / "metadata.json").exists() and p.name != "_reference":
                continue
            out.append(p.name)
        return out

    def lifecycle(self, name: str) -> Lifecycle:
        meta = self._read_metadata(name)
        return Lifecycle(meta.get("lifecycle", Lifecycle.GENERATED.value))

    def history(self, name: str) -> list[ValidationRun]:
        path = self.root / name / "validation.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        runs = data.get("runs", [])
        return [ValidationRun(**r) for r in runs]

    def record(self, name: str) -> DialectRecord:
        meta_dict = self._read_metadata(name)
        history = self.history(name)
        return DialectRecord(
            name=name,
            lifecycle=Lifecycle(meta_dict.get("lifecycle", Lifecycle.GENERATED.value)),
            metadata=GenerationMetadata.from_dict(meta_dict.get("generation", {})),
            last_validation=history[-1] if history else None,
        )

    def load(self, name: str) -> DialectEngine:
        d = self.root / name
        if not d.exists():
            raise FileNotFoundError(f"Dialect not found: {name} (looking in {d})")

        grammar_text = (d / "grammar.lark").read_text()
        spec = (
            json.loads((d / "spec.json").read_text())
            if (d / "spec.json").exists()
            else {}
        )
        semantics = SemanticConfig.model_validate_json(
            (d / "semantics.json").read_text()
        )

        lowering_mod = self._load_module(d / "lowering.py", f"manysql._loaded.{name}.lowering")
        overrides_mod = None
        if (d / "overrides.py").exists():
            overrides_mod = self._load_module(
                d / "overrides.py", f"manysql._loaded.{name}.overrides"
            )
        passes_mod = None
        if (d / "passes.py").exists():
            passes_mod = self._load_module(
                d / "passes.py", f"manysql._loaded.{name}.passes"
            )
        effects_mod = None
        if (d / "effects.py").exists():
            effects_mod = self._load_module(
                d / "effects.py", f"manysql._loaded.{name}.effects"
            )

        meta_dict = self._read_metadata(name)
        return DialectEngine(
            name=name,
            spec=spec,
            grammar_text=grammar_text,
            lowering=lowering_mod,
            semantics=semantics,
            overrides=overrides_mod,
            passes=passes_mod,
            effects=effects_mod,
            metadata=GenerationMetadata.from_dict(meta_dict.get("generation", {})),
            lifecycle=Lifecycle(meta_dict.get("lifecycle", Lifecycle.GENERATED.value)),
        )

    def save(
        self,
        name: str,
        *,
        spec: dict[str, Any],
        grammar_text: str,
        lowering_source: str,
        semantics: SemanticConfig,
        overrides_source: Optional[str] = None,
        passes_source: Optional[str] = None,
        effects_source: Optional[str] = None,
        generation: Optional[GenerationMetadata] = None,
        lifecycle: Lifecycle = Lifecycle.GENERATED,
    ) -> Path:
        if name in RESERVED_NAMES:
            raise ValueError(f"Reserved dialect name: {name}")
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")
        (d / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True))
        (d / "grammar.lark").write_text(grammar_text)
        (d / "lowering.py").write_text(lowering_source)
        (d / "semantics.json").write_text(
            semantics.model_dump_json(indent=2, exclude_none=False)
        )
        if overrides_source is not None:
            (d / "overrides.py").write_text(overrides_source)
        if passes_source is not None:
            (d / "passes.py").write_text(passes_source)
        if effects_source is not None:
            (d / "effects.py").write_text(effects_source)

        gen = generation or GenerationMetadata()
        gen.updated_at = datetime.now(timezone.utc).isoformat()
        meta = {"lifecycle": lifecycle.value, "generation": gen.to_dict()}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        return d

    def set_lifecycle(self, name: str, lifecycle: Lifecycle) -> None:
        meta = self._read_metadata(name)
        meta["lifecycle"] = lifecycle.value
        gen = GenerationMetadata.from_dict(meta.get("generation", {}))
        gen.updated_at = datetime.now(timezone.utc).isoformat()
        meta["generation"] = gen.to_dict()
        (self.root / name / "metadata.json").write_text(json.dumps(meta, indent=2))

    def append_validation(self, name: str, run: ValidationRun) -> None:
        path = self.root / name / "validation.json"
        data = {"runs": []}
        if path.exists():
            data = json.loads(path.read_text())
        data["runs"].append(run.to_dict())
        path.write_text(json.dumps(data, indent=2, default=str))

    # --- internals ---

    def _read_metadata(self, name: str) -> dict[str, Any]:
        path = self.root / name / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    @staticmethod
    def _load_module(path: Path, module_name: str) -> ModuleType:
        import sys

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        # Register in sys.modules before exec_module so dataclasses (and other
        # importlib-aware machinery) can look the module up by name during init.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module
