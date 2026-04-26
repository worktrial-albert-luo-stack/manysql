"""Hand-curated golden queries against the test catalog.

Categories:
    scan: bare table reads
    filter: WHERE-clause predicates exercising each operator
    project: SELECT-list expressions (arithmetic, string, case, cast)
    join: every join kind, with ON / USING / multi-condition
    aggregate: GROUP BY shapes, every agg function, HAVING, DISTINCT
    sort_limit: ORDER BY directions / nulls placement, LIMIT/OFFSET
    distinct: SELECT DISTINCT
    set_op: UNION / INTERSECT / EXCEPT (DISTINCT and ALL)
    cte: WITH bindings, multi-binding, dependent CTEs
    subquery: scalar / EXISTS / IN / NOT IN, FROM-subquery
    window: ROW_NUMBER / RANK / DENSE_RANK / SUM / AVG / LAG / LEAD
    semantic: queries whose result is sensitive to a SemanticConfig knob

Each entry has a `cross_dialect` flag indicating whether the SQL surface is
plain enough to re-render across dialects (most are).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    category: str
    sql: str
    cross_dialect: bool = True
    notes: str = ""


GOLDEN_QUERIES: list[GoldenQuery] = [
    # ---- scan ----
    GoldenQuery("scan_employees", "scan", "SELECT * FROM employees"),
    GoldenQuery("scan_departments", "scan", "SELECT * FROM departments"),
    GoldenQuery("scan_sales", "scan", "SELECT * FROM sales"),
    GoldenQuery("scan_subset", "scan", "SELECT id, name FROM employees"),

    # ---- filter ----
    GoldenQuery("filter_eq", "filter", "SELECT id FROM employees WHERE dept_id = 10"),
    GoldenQuery("filter_neq", "filter", "SELECT id FROM employees WHERE dept_id <> 10"),
    GoldenQuery("filter_lt", "filter", "SELECT id FROM employees WHERE salary < 90000"),
    GoldenQuery("filter_gte", "filter", "SELECT id FROM employees WHERE salary >= 100000"),
    GoldenQuery("filter_and", "filter", "SELECT id FROM employees WHERE salary > 80000 AND active"),
    GoldenQuery("filter_or", "filter", "SELECT id FROM employees WHERE dept_id = 10 OR dept_id = 20"),
    GoldenQuery("filter_not", "filter", "SELECT id FROM employees WHERE NOT active"),
    GoldenQuery("filter_in_list", "filter", "SELECT id FROM employees WHERE dept_id IN (10, 30)"),
    GoldenQuery("filter_not_in_list", "filter", "SELECT id FROM employees WHERE dept_id NOT IN (10)"),
    GoldenQuery("filter_between", "filter", "SELECT id FROM employees WHERE salary BETWEEN 80000 AND 120000"),
    GoldenQuery("filter_not_between", "filter", "SELECT id FROM employees WHERE salary NOT BETWEEN 80000 AND 120000"),
    GoldenQuery("filter_is_null", "filter", "SELECT id FROM employees WHERE dept_id IS NULL"),
    GoldenQuery("filter_is_not_null", "filter", "SELECT id FROM employees WHERE dept_id IS NOT NULL"),
    GoldenQuery("filter_like", "filter", "SELECT id FROM employees WHERE name LIKE 'A%'"),
    GoldenQuery(
        "filter_compound",
        "filter",
        "SELECT id FROM employees WHERE (active AND salary > 90000) OR dept_id IS NULL",
    ),

    # ---- project ----
    GoldenQuery(
        "project_arithmetic",
        "project",
        "SELECT id, salary * 0.1 AS bonus FROM employees",
    ),
    GoldenQuery(
        "project_arith_chain",
        "project",
        "SELECT id, (salary + 1000) * 1.05 AS adj FROM employees",
    ),
    GoldenQuery(
        "project_concat",
        "project",
        "SELECT id, name || ' (active)' AS tag FROM employees WHERE active",
    ),
    GoldenQuery(
        "project_case",
        "project",
        "SELECT id, CASE WHEN salary > 100000 THEN 'high' WHEN salary > 80000 THEN 'mid' ELSE 'low' END AS bucket FROM employees",
    ),
    GoldenQuery(
        "project_cast_int_float",
        "project",
        "SELECT id, CAST(salary AS BIGINT) AS rounded FROM employees",
    ),
    GoldenQuery(
        "project_neg",
        "project",
        "SELECT id, -salary AS neg FROM employees",
    ),

    # ---- join ----
    GoldenQuery(
        "join_inner_on",
        "join",
        "SELECT e.id, d.name FROM employees e INNER JOIN departments d ON e.dept_id = d.id",
    ),
    GoldenQuery(
        "join_left_on",
        "join",
        "SELECT e.id, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id",
    ),
    GoldenQuery(
        "join_right_on",
        "join",
        "SELECT e.id, d.name FROM employees e RIGHT JOIN departments d ON e.dept_id = d.id",
    ),
    GoldenQuery(
        "join_full_on",
        "join",
        "SELECT e.id, d.name FROM employees e FULL OUTER JOIN departments d ON e.dept_id = d.id",
        cross_dialect=False,
        notes="FULL OUTER JOIN not supported by all engines",
    ),
    GoldenQuery(
        "join_cross",
        "join",
        "SELECT e.id, d.id AS d_id FROM employees e CROSS JOIN departments d",
    ),
    GoldenQuery(
        "join_multi_cond",
        "join",
        "SELECT e.id FROM employees e INNER JOIN departments d "
        "ON e.dept_id = d.id AND d.budget > 500000",
    ),
    GoldenQuery(
        "join_three_way",
        "join",
        "SELECT e.id, d.name AS dept, s.amount "
        "FROM employees e INNER JOIN departments d ON e.dept_id = d.id "
        "INNER JOIN sales s ON s.employee_id = e.id",
    ),

    # ---- aggregate ----
    GoldenQuery(
        "agg_count_star",
        "aggregate",
        "SELECT COUNT(*) AS n FROM employees",
    ),
    GoldenQuery(
        "agg_count_col",
        "aggregate",
        "SELECT COUNT(dept_id) AS n FROM employees",
    ),
    GoldenQuery(
        "agg_sum_avg_min_max",
        "aggregate",
        "SELECT SUM(salary) AS s, AVG(salary) AS a, MIN(salary) AS lo, MAX(salary) AS hi FROM employees",
    ),
    GoldenQuery(
        "agg_group_by",
        "aggregate",
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id",
    ),
    GoldenQuery(
        "agg_group_by_two_keys",
        "aggregate",
        "SELECT dept_id, active, COUNT(*) AS n FROM employees GROUP BY dept_id, active",
    ),
    GoldenQuery(
        "agg_having",
        "aggregate",
        "SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1",
    ),
    GoldenQuery(
        "agg_count_distinct",
        "aggregate",
        "SELECT COUNT(DISTINCT dept_id) AS unique_depts FROM employees",
    ),
    GoldenQuery(
        "agg_filtered",
        "aggregate",
        "SELECT dept_id, SUM(salary) FILTER (WHERE active) AS active_pay FROM employees GROUP BY dept_id",
        cross_dialect=False,
        notes="FILTER clause not universally supported",
    ),

    # ---- sort / limit ----
    GoldenQuery(
        "sort_asc",
        "sort_limit",
        "SELECT id FROM employees ORDER BY salary ASC",
    ),
    GoldenQuery(
        "sort_desc_nulls_last",
        "sort_limit",
        "SELECT id FROM employees ORDER BY dept_id DESC NULLS LAST",
    ),
    GoldenQuery(
        "sort_multi_key",
        "sort_limit",
        "SELECT id FROM employees ORDER BY active DESC, salary ASC, id ASC",
    ),
    GoldenQuery(
        "sort_limit",
        "sort_limit",
        "SELECT id FROM employees ORDER BY salary DESC LIMIT 5",
    ),
    GoldenQuery(
        "sort_limit_offset",
        "sort_limit",
        "SELECT id FROM employees ORDER BY salary DESC LIMIT 3 OFFSET 2",
    ),

    # ---- distinct ----
    GoldenQuery(
        "distinct_one_col",
        "distinct",
        "SELECT DISTINCT dept_id FROM employees",
    ),
    GoldenQuery(
        "distinct_multi_col",
        "distinct",
        "SELECT DISTINCT dept_id, active FROM employees",
    ),

    # ---- set ops ----
    GoldenQuery(
        "union_distinct",
        "set_op",
        "SELECT id FROM employees UNION SELECT id FROM departments",
    ),
    GoldenQuery(
        "union_all",
        "set_op",
        "SELECT id FROM employees UNION ALL SELECT id FROM departments",
    ),
    GoldenQuery(
        "intersect",
        "set_op",
        "SELECT dept_id FROM employees INTERSECT SELECT id FROM departments",
    ),
    GoldenQuery(
        "except_op",
        "set_op",
        "SELECT id FROM departments EXCEPT SELECT dept_id FROM employees",
    ),

    # ---- CTE ----
    GoldenQuery(
        "cte_simple",
        "cte",
        "WITH active_emps AS (SELECT id, name FROM employees WHERE active) "
        "SELECT name FROM active_emps",
    ),
    GoldenQuery(
        "cte_multi",
        "cte",
        "WITH high_pay AS (SELECT id FROM employees WHERE salary > 100000), "
        "active_emps AS (SELECT id FROM employees WHERE active) "
        "SELECT id FROM high_pay UNION SELECT id FROM active_emps",
    ),
    GoldenQuery(
        "cte_referenced_twice",
        "cte",
        "WITH e AS (SELECT id, dept_id, salary FROM employees) "
        "SELECT a.id FROM e a INNER JOIN e b ON a.dept_id = b.dept_id AND a.salary > b.salary",
    ),

    # ---- subquery ----
    GoldenQuery(
        "subq_scalar",
        "subquery",
        "SELECT id, salary - (SELECT AVG(salary) FROM employees) AS gap FROM employees",
    ),
    GoldenQuery(
        "subq_in_uncorrelated",
        "subquery",
        "SELECT name FROM employees "
        "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 500000)",
    ),
    GoldenQuery(
        "subq_not_in_uncorrelated",
        "subquery",
        "SELECT name FROM employees "
        "WHERE dept_id NOT IN (SELECT id FROM departments WHERE budget < 200000)",
    ),
    GoldenQuery(
        "subq_exists_uncorrelated",
        "subquery",
        "SELECT name FROM employees "
        "WHERE EXISTS (SELECT 1 FROM departments WHERE budget > 1000000)",
    ),
    GoldenQuery(
        "subq_from",
        "subquery",
        "SELECT t.id FROM (SELECT id, salary FROM employees WHERE active) AS t WHERE t.salary > 90000",
    ),

    # ---- window ----
    GoldenQuery(
        "window_row_number_partitioned",
        "window",
        "SELECT id, dept_id, "
        "ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn "
        "FROM employees",
    ),
    GoldenQuery(
        "window_rank",
        "window",
        "SELECT id, RANK() OVER (ORDER BY salary DESC) AS r FROM employees",
    ),
    GoldenQuery(
        "window_dense_rank",
        "window",
        "SELECT id, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr FROM employees",
    ),
    GoldenQuery(
        "window_running_sum",
        "window",
        "SELECT id, dept_id, "
        "SUM(salary) OVER (PARTITION BY dept_id ORDER BY id) AS running "
        "FROM employees",
    ),

    # ---- semantic-knob-sensitive ----
    GoldenQuery(
        "semantic_div_by_zero",
        "semantic",
        "SELECT id, salary / 0.0 AS x FROM employees LIMIT 1",
        notes="Reference dialect: division_by_zero=NULL",
    ),
    GoldenQuery(
        "semantic_null_order_asc_default",
        "semantic",
        "SELECT id, dept_id FROM employees ORDER BY dept_id ASC",
        notes="Reference dialect: null_order_default_asc=LAST",
    ),
]


__all__ = ["GoldenQuery", "GOLDEN_QUERIES"]
