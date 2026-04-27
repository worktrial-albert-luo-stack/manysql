"""Sandboxed loader for a dialect's `overrides.py` / `passes.py` / `effects.py`.

Why sandbox? Generated dialect modules come either from an LLM (untrusted) or a
human author (trusted, but still nice to fail loudly on accidents). The
loader:

  - Restricts `__import__` to a small allowlist (polars, pyarrow, math,
    typing, the manysql IR & spec modules, etc.). Any other import raises
    `OverrideImportError` at load time.
  - Strips dangerous builtins (`open`, `eval`, `exec`, `compile`, `__import__`).
  - Compiles the source with the supplied filename so tracebacks are
    readable.

The loader returns a plain `ModuleType` whose public dicts / lists the
runtime can consult.

`load_overrides` is the historical entry point and the default for
`overrides.py`. `load_sandboxed_module` is a generic alias for callers
loading `passes.py` or `effects.py`; both share the allowlist, since
passes need to construct IR nodes and effects need IR-typed signatures.

This sandbox is intentionally conservative: it stops accidents and
trivial misuse. It does not defend against a determined adversary inside
the same Python process. For training / dataset curation it is enough;
for production use, run inside a separate subprocess with OS-level
isolation.
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
from typing import Optional


_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "polars",
        "pyarrow",
        "math",
        "datetime",
        "decimal",
        "fractions",
        "statistics",
        "typing",
        "collections",
        "collections.abc",
        "functools",
        "itertools",
        "re",
        "json",
        "operator",
        "dataclasses",
        "manysql.spec.semantics",
        "manysql.ir.expr",
        "manysql.ir.plan",
        "manysql.ir.types",
    }
)


class OverrideImportError(RuntimeError):
    """Raised when a dialect-extension module attempts a disallowed import."""


def load_sandboxed_module(
    source: str,
    *,
    fullname: str,
    extra_allowed: Optional[frozenset[str]] = None,
) -> ModuleType:
    """Compile and execute `source` with a restricted import allowlist.

    Used by the dialect registry to load any of `overrides.py`, `passes.py`,
    `effects.py` from a per-dialect package. The returned module is
    registered in `sys.modules[fullname]` so that type-annotation and
    dataclass machinery can find it; callers should pass unique fullnames
    (typically `f"manysql._loaded.{dialect}.{kind}"`) to avoid collisions
    across dialects.
    """
    allowed = _ALLOWED_IMPORTS | (extra_allowed or frozenset())

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[index]

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ARG001
        root = name.split(".")[0]
        if name in allowed or root in allowed or _is_subpath_allowed(name, allowed):
            return real_import(name, globals, locals, fromlist, level)
        raise OverrideImportError(
            f"override module attempted disallowed import: {name!r}"
        )

    safe_builtins = {
        k: v
        for k, v in (
            __builtins__.items() if isinstance(__builtins__, dict) else vars(__builtins__).items()
        )
        if k not in _BLOCKED_BUILTINS
    }
    safe_builtins["__import__"] = _restricted_import

    class _Loader:
        def create_module(self, spec):  # noqa: ARG002
            return None

        def exec_module(self, module):  # noqa: D401
            module.__dict__["__builtins__"] = safe_builtins
            exec(compile(source, fullname, "exec"), module.__dict__)

    spec = spec_from_loader(fullname, _Loader())
    if spec is None:
        raise OverrideImportError(f"could not build module spec for {fullname}")
    module = module_from_spec(spec)
    sys.modules[fullname] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(fullname, None)
        raise
    return module


def _is_subpath_allowed(name: str, allowed: frozenset[str]) -> bool:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        if ".".join(parts[:i]) in allowed:
            return True
    return False


_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "exit",
        "quit",
        "breakpoint",
        "help",
    }
)


def load_overrides(
    source: str,
    *,
    fullname: str = "_codegen_overrides",
    extra_allowed: Optional[frozenset[str]] = None,
) -> ModuleType:
    """Backwards-compatible alias for `load_sandboxed_module`.

    Existing callers that load `overrides.py` keep working unchanged. New
    callers loading `passes.py` / `effects.py` should prefer
    `load_sandboxed_module` for clarity.
    """
    return load_sandboxed_module(
        source, fullname=fullname, extra_allowed=extra_allowed
    )


__all__ = [
    "OverrideImportError",
    "load_overrides",
    "load_sandboxed_module",
]
