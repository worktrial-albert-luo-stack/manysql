"""Typed-value model for the IR.

Kept deliberately small for v1 (Tier A scope):
    INT, FLOAT, TEXT, BOOL, DATE, TIMESTAMP, NULL.

Tier B (arrays, structs, maps, JSON) will extend this module via the IR-extension
RFC process documented in manysql/ir/SCOPE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Union


class TypeKind(str, Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    TEXT = "TEXT"
    BOOL = "BOOL"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    NULL = "NULL"


@dataclass(frozen=True)
class IRType:
    kind: TypeKind
    nullable: bool = True

    def __str__(self) -> str:
        return f"{self.kind.value}{'?' if self.nullable else ''}"


INT = IRType(TypeKind.INT)
FLOAT = IRType(TypeKind.FLOAT)
TEXT = IRType(TypeKind.TEXT)
BOOL = IRType(TypeKind.BOOL)
DATE_T = IRType(TypeKind.DATE)
TIMESTAMP = IRType(TypeKind.TIMESTAMP)
NULL = IRType(TypeKind.NULL, nullable=True)


PyValue = Union[int, float, str, bool, date, datetime, None]
