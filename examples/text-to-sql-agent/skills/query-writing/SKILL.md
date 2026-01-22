---
name: query-writing
description: For writing and executing SQL queries - from simple single-table queries to complex multi-table JOINs and aggregations
---

# Query Writing Skill

## When to Use This Skill

Use this skill when you need to answer a question by writing and executing a SQL query.

## Workflow for Simple Queries

For straightforward questions about a single table:

1. **Identify the table** - Which table has the data?
2. **Get the schema** - Use `sql_db_schema` to see columns
3. **Write the query** - SELECT relevant columns with WHERE/LIMIT/ORDER BY
4. **Execute** - Run with `sql_db_query`
5. **Format answer** - Present results clearly

## Workflow for Complex Queries

For questions requiring multiple tables:

### 1. Plan Your Approach
**Use `write_todos` to break down the task:**
- Identify all tables needed
- Map relationships (foreign keys)
- Plan JOIN structure
- Determine aggregations

### 2. Examine Schemas
Use `sql_db_schema` for EACH table to find join columns and needed fields.

### 3. Construct Query
- SELECT - Columns and aggregates
- FROM/JOIN - Connect tables on FK = PK
- WHERE - Filters before aggregation
- GROUP BY - All non-aggregate columns
- ORDER BY - Sort meaningfully
- LIMIT - Default 5 rows

### 4. Validate and Execute
Check all JOINs have conditions, GROUP BY is correct, then run query.

## Example: Revenue by Country
```sql
SELECT
    c.Country,
    ROUND(SUM(i.Total), 2) as TotalRevenue
FROM Invoice i
INNER JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalRevenue DESC
LIMIT 5;
```

## Quality Guidelines

- Query only relevant columns (not SELECT *)
- Always apply LIMIT (5 default)
- Use table aliases for clarity
- For complex queries: use write_todos to plan
- Never use DML statements (INSERT, UPDATE, DELETE, DROP)
