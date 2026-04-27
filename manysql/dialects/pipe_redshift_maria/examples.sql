-- manysql-codegen examples for dialect: pipe_redshift_maria
-- Hand-curated canonical SQL queries rewritten into this dialect's surface.
-- These are the same items used by the parse and IR-equivalence batteries.
-- Re-generate with: manysql-codegen gen pipe_redshift_maria --overwrite

-- scan_all
FETCH * FROM employees;

-- scan_subset
FETCH id, name FROM employees;

-- filter_eq
FETCH id FROM employees FILTER dept_id = 10;

-- filter_neq
FETCH id FROM employees FILTER dept_id <> 10;

-- filter_in
FETCH id FROM employees FILTER dept_id IN (10, 20);

-- filter_between
FETCH id FROM employees FILTER salary BETWEEN 80000 AND 120000;

-- filter_like
FETCH id FROM employees FILTER name LIKE 'A%';

-- filter_is_null
FETCH id FROM employees FILTER dept_id IS NONE;

-- filter_is_not_null
FETCH id FROM employees FILTER dept_id IS NOT NONE;

-- project_arith
FETCH id, salary + 1000 AS bumped FROM employees;

-- project_concat
FETCH name || '!' AS shouted FROM employees;

-- project_case
FETCH id, CASE WHEN salary > 100000 THEN 'high' ELSE 'low' END AS tier FROM employees;

-- join_inner
FETCH e.id, d.name FROM employees e JOIN departments d ON e.dept_id = d.id;

-- join_left
FETCH e.id, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id;

-- agg_group_by
FETCH dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id;

-- agg_having
FETCH dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id HAVING COUNT(*) > 1;

-- order_asc
FETCH id FROM employees ARRANGE BY salary ASC;

-- order_desc_nulls_last
FETCH id FROM employees ARRANGE BY dept_id DESC NULLS LAST;

-- limit_only
FETCH id FROM employees LIMIT 5;

-- limit_offset
FETCH id FROM employees LIMIT 5 OFFSET 10;

-- distinct
FETCH UNIQUE dept_id FROM employees;

-- union_all
FETCH id FROM employees UNION ALL FETCH id FROM departments;

-- subq_in
FETCH id FROM employees FILTER dept_id IN (FETCH id FROM departments);

-- cte_simple
WITH high AS (FETCH id FROM employees FILTER salary > 100000) FETCH * FROM high;

-- window_row_number
FETCH id, ROW_NUMBER() OVER (PARTITION BY dept_id ARRANGE BY salary) AS rn FROM employees;

-- cast_int
FETCH CAST(salary AS INT) FROM employees;
