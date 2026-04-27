-- manysql-codegen examples for dialect: redshift_oracle_mysql_convert
-- Hand-curated canonical SQL queries rewritten into this dialect's surface.
-- These are the same items used by the parse and IR-equivalence batteries.
-- Re-generate with: manysql-codegen gen redshift_oracle_mysql_convert --overwrite

-- scan_all
SELECT * FROM employees

-- scan_subset
SELECT id, name FROM employees

-- filter_eq
SELECT id FROM employees WHERE dept_id = 10

-- filter_neq
SELECT id FROM employees WHERE dept_id <> 10

-- filter_in
SELECT id FROM employees WHERE dept_id IN (10, 20)

-- filter_between
SELECT id FROM employees WHERE salary BETWEEN 80000 AND 120000

-- filter_like
SELECT id FROM employees WHERE name LIKE 'A%'

-- filter_is_null
SELECT id FROM employees WHERE dept_id IS NULL

-- filter_is_not_null
SELECT id FROM employees WHERE dept_id IS NOT NULL

-- project_arith
SELECT id, salary + 1000 AS bumped FROM employees

-- project_concat
SELECT name || '!' AS shouted FROM employees

-- project_case
SELECT id, CASE WHEN salary > 100000 THEN 'high' ELSE 'low' END AS tier FROM employees

-- join_inner
SELECT e.id, d.name FROM employees e INNER JOIN departments d ON e.dept_id = d.id

-- join_left
SELECT e.id, d.name FROM employees e LEFT OUTER JOIN departments d ON e.dept_id = d.id

-- agg_group_by
SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id

-- agg_having
SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1

-- order_asc
SELECT id FROM employees ORDER BY salary ASC

-- order_desc_nulls_last
SELECT id FROM employees ORDER BY dept_id DESC NULLS LAST

-- limit_only
SELECT id FROM employees LIMIT 5

-- limit_offset
SELECT id FROM employees LIMIT 5 OFFSET 10

-- distinct
SELECT DISTINCT dept_id FROM employees

-- union_all
SELECT id FROM employees UNION ALL SELECT id FROM departments

-- subq_in
SELECT id FROM employees WHERE dept_id IN (SELECT id FROM departments)

-- cte_simple
WITH high AS (SELECT id FROM employees WHERE salary > 100000) SELECT * FROM high

-- window_row_number
SELECT id, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary) AS rn FROM employees

-- cast_int
SELECT CAST(salary AS INT) FROM employees
