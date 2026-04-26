"""Catalog: schemas for the test datasets used by golden plans and oracles.

Tables are designed to exercise:
- nullables in every column type
- ties (multiple rows with identical sort keys)
- date-arithmetic edge cases (month boundaries, leap years)
- correlated-subquery friendly shapes (employees / departments / regions)
- recursive-CTE shape (orgchart / categories tree)
- group-by with empty groups, ROLLUP-like cases

Datasets are stored as Parquet under manysql/storage/data/ and regenerated
deterministically from `seed_datasets()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from manysql.ir.plan import ColumnSchema
from manysql.ir.types import BOOL, DATE_T, FLOAT, INT, TEXT, IRType

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class TableMeta:
    name: str
    columns: tuple[ColumnSchema, ...]


def _col(name: str, t: IRType) -> ColumnSchema:
    return ColumnSchema(name=name, type=t)


CATALOG: dict[str, TableMeta] = {
    "employees": TableMeta(
        name="employees",
        columns=(
            _col("id", INT),
            _col("name", TEXT),
            _col("dept_id", INT),
            _col("manager_id", INT),  # nullable for org root
            _col("salary", FLOAT),
            _col("hired_on", DATE_T),
            _col("active", BOOL),
        ),
    ),
    "departments": TableMeta(
        name="departments",
        columns=(
            _col("id", INT),
            _col("name", TEXT),
            _col("region_id", INT),
            _col("budget", FLOAT),
        ),
    ),
    "regions": TableMeta(
        name="regions",
        columns=(
            _col("id", INT),
            _col("name", TEXT),
        ),
    ),
    "sales": TableMeta(
        name="sales",
        columns=(
            _col("id", INT),
            _col("employee_id", INT),
            _col("amount", FLOAT),
            _col("sold_on", DATE_T),
            _col("region_id", INT),  # nullable: some sales unallocated
        ),
    ),
    # For recursive CTEs: simple categories tree
    "categories": TableMeta(
        name="categories",
        columns=(
            _col("id", INT),
            _col("name", TEXT),
            _col("parent_id", INT),  # nullable: root category has none
        ),
    ),
}


def seed_datasets() -> dict[str, pl.DataFrame]:
    """Build the canonical test datasets in-memory.

    Deterministic: same data every call. Uses Polars-native types matching the
    catalog's IRType declarations.
    """

    employees = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "name": ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"],
            "dept_id": [10, 10, 20, 20, 30, 30, 30, None],  # heidi unassigned
            "manager_id": [None, 1, None, 3, 3, 5, 5, 5],
            "salary": [120000.0, 90000.0, 150000.0, 80000.0, 110000.0, 70000.0, None, 95000.0],
            "hired_on": [
                date(2018, 1, 15),
                date(2019, 6, 1),
                date(2017, 3, 20),
                date(2020, 11, 30),
                date(2016, 7, 4),
                date(2021, 2, 14),
                date(2022, 8, 8),
                date(2024, 2, 29),  # leap day
            ],
            "active": [True, True, True, False, True, True, True, True],
        },
        schema={
            "id": pl.Int64,
            "name": pl.Utf8,
            "dept_id": pl.Int64,
            "manager_id": pl.Int64,
            "salary": pl.Float64,
            "hired_on": pl.Date,
            "active": pl.Boolean,
        },
    )

    departments = pl.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "name": ["Engineering", "Sales", "Research", "Marketing"],
            "region_id": [1, 2, 1, None],  # Marketing has no region (LEFT JOIN test)
            "budget": [1_000_000.0, 800_000.0, 1_500_000.0, 200_000.0],
        },
        schema={
            "id": pl.Int64,
            "name": pl.Utf8,
            "region_id": pl.Int64,
            "budget": pl.Float64,
        },
    )

    regions = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["NA", "EU", "APAC"],  # Region 3 has no department: RIGHT/FULL JOIN test
        },
        schema={"id": pl.Int64, "name": pl.Utf8},
    )

    sales = pl.DataFrame(
        {
            "id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "employee_id": [1, 1, 2, 2, 3, 3, 5, 5, 5, 6],
            "amount": [100.0, 200.0, 50.0, 50.0, 1000.0, None, 75.0, 75.0, 25.0, 0.0],
            "sold_on": [
                date(2024, 1, 31),
                date(2024, 2, 29),  # leap day
                date(2024, 3, 1),
                date(2024, 3, 31),
                date(2024, 12, 31),
                date(2024, 6, 15),
                date(2024, 7, 4),
                date(2024, 7, 4),  # tie on date
                date(2024, 7, 5),
                date(2024, 8, 1),
            ],
            "region_id": [1, 1, 2, 2, 1, 1, None, None, 1, 2],
        },
        schema={
            "id": pl.Int64,
            "employee_id": pl.Int64,
            "amount": pl.Float64,
            "sold_on": pl.Date,
            "region_id": pl.Int64,
        },
    )

    # Categories tree:
    #   1 root
    #   ├─ 2 electronics
    #   │  ├─ 4 phones
    #   │  └─ 5 computers
    #   │      └─ 7 laptops
    #   └─ 3 books
    #      └─ 6 fiction
    categories = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "name": ["root", "electronics", "books", "phones", "computers", "fiction", "laptops"],
            "parent_id": [None, 1, 1, 2, 2, 3, 5],
        },
        schema={"id": pl.Int64, "name": pl.Utf8, "parent_id": pl.Int64},
    )

    return {
        "employees": employees,
        "departments": departments,
        "regions": regions,
        "sales": sales,
        "categories": categories,
    }


def materialize() -> None:
    """Write all seed datasets to Parquet files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in seed_datasets().items():
        df.write_parquet(DATA_DIR / f"{name}.parquet")


def load_table(name: str) -> pl.DataFrame:
    """Load a table from disk (materializing first if needed)."""
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        materialize()
    return pl.read_parquet(p)


def load_all() -> dict[str, pl.DataFrame]:
    """Load every table in the catalog."""
    return {name: load_table(name) for name in CATALOG}


def schema_of(name: str) -> tuple[ColumnSchema, ...]:
    """Look up a table's IR schema."""
    if name not in CATALOG:
        raise KeyError(f"Unknown table: {name}")
    return CATALOG[name].columns


def to_dict_for_oracle(tables: dict[str, pl.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    """Convert tables to row-dict form, used by SQL-engine oracles to load data."""
    return {name: df.to_dicts() for name, df in tables.items()}
