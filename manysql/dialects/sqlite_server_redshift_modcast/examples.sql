-- manysql-codegen examples for dialect: sqlite_server_redshift_modcast
-- Hand-curated canonical SQL queries rewritten into this dialect's surface.
-- These are the same items used by the parse and IR-equivalence batteries.
-- Re-generate with: manysql-codegen gen sqlite_server_redshift_modcast --overwrite

-- scan_all
RETRIEVE * FROM employees

-- scan_subset
RETRIEVE id, name FROM employees

-- filter_eq
RETRIEVE id FROM employees WHERE dept_id = 10

-- filter_neq
RETRIEVE id FROM employees WHERE dept_id <> 10

-- filter_in
RETRIEVE id FROM employees WHERE dept_id IN (10, 20)

-- filter_between
RETRIEVE id FROM employees WHERE salary BETWEEN 80000 AND 120000

-- filter_like
RETRIEVE id FROM employees WHERE name LIKE 'A%'

-- filter_is_null
RETRIEVE id FROM employees WHERE dept_id IS NULL

-- filter_is_not_null
RETRIEVE id FROM employees WHERE dept_id IS NOT NULL

-- project_arith
RETRIEVE id, salary + 1000 AS bumped FROM employees

-- project_concat
RETRIEVE name || '!' AS shouted FROM employees

-- project_case
RETRIEVE id, CASE WHEN salary > 100000 THEN 'high' ELSE 'low' END AS tier FROM employees

-- join_inner
RETRIEVE e.id, d.name FROM employees e INNER JOIN departments d ON e.dept_id = d.id

-- join_left
RETRIEVE e.id, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id

-- agg_group_by
RETRIEVE dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id

-- agg_having
RETRIEVE dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1

-- order_asc
RETRIEVE id FROM employees ORDER BY salary ASC

-- order_desc_nulls_last
RETRIEVE id FROM employees ORDER BY dept_id DESC NULLS LAST

-- limit_only
RETRIEVE id FROM employees LIMIT 5

-- limit_offset
RETRIEVE id FROM employees LIMIT 5 OFFSET 10

-- distinct
RETRIEVE UNIQUE dept_id FROM employees

-- union_all
RETRIEVE id FROM employees UNION ALL RETRIEVE id FROM departments

-- subq_in
RETRIEVE id FROM employees WHERE dept_id IN (RETRIEVE id FROM departments)

-- cte_simple
WITH high AS (RETRIEVE id FROM employees WHERE salary > 100000) RETRIEVE * FROM high

-- window_row_number
RETRIEVE id, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary) AS rn FROM employees

-- cast_int
RETRIEVE CAST(salary AS INT) FROM employees
