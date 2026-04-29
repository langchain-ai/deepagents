---
name: exploratory-data-analysis
description: Inspect, profile, and validate one or more data files before answering analytical questions or creating reports.
---

# Exploratory Data Analysis

Use this skill when you need to understand one or more data files before answering analytical questions.

## Workflow

1. Discover candidate files in `/memories/skills/data/` and select the files relevant to the user's question.
2. Inspect each selected file's path, format, header/schema, row count, and sample rows.
3. Load the data with Python and print column names, dtypes, missing counts, and basic descriptive statistics.
4. Check for duplicate rows and obvious invalid values, such as negative counts, impossible rates, out-of-range values, or dates that fail to parse.
5. Identify useful dimensions, measures, time columns, and possible entity keys.
6. For multiple files, inspect schemas separately before joining or concatenating; validate row counts and join cardinality.
7. Summarize data quality before moving into conclusions.

## Checks To Run

- `df.shape`
- `df.head()`
- `df.info()`
- `df.isna().sum()`
- `df.describe(include="all")`
- duplicate row count
- date parsing success for date-like columns
- unique counts for categorical columns
- file-level schema comparison when multiple files are selected
- join key uniqueness and unmatched row counts when combining files

## Output

Report the files analyzed, dataset shapes, key columns, potential metrics, data quality issues, and the next analysis steps. Do not present business conclusions until you have validated the relevant fields and any multi-file relationships.
