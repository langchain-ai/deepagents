---
name: statistical-summary
description: Use this skill for targeted statistical analysis — group-by aggregations, top-N rankings, time-series trends, and pivot tables
---

# Statistical Summary Skill

This skill provides a workflow for answering specific analytical questions about a dataset using aggregations, rankings, and trend analysis.

## When to Use This Skill

Use this skill when asked to:
- Compare categories (e.g., "Which region has the highest revenue?")
- Rank items (e.g., "Top 5 products by quantity sold")
- Analyze trends over time (e.g., "Monthly revenue trend")
- Compute aggregations (e.g., "Average order value by customer")
- Build pivot tables or cross-tabulations

## Workflow

### 1. Understand the Question

Identify:
- **Metric**: What is being measured (revenue, quantity, count, average, etc.)
- **Dimension**: What is being grouped by (region, product, month, customer, etc.)
- **Filter**: Any conditions to apply (date range, specific category, etc.)
- **Ordering**: How results should be sorted (top-N, ascending, descending)

### 2. Plan the Script

Use `write_todos` for multi-step analyses. For each step, decide:
- Which columns to use
- What aggregation to apply (`sum`, `mean`, `count`, `median`, etc.)
- Whether to parse dates with `pd.to_datetime()`
- Whether a chart would help communicate the result

### 3. Run Aggregations

Execute self-contained pandas scripts via the `execute` tool. Examples:

**Group-by with sorting:**
```python
import pandas as pd
df = pd.read_csv('<csv_path>')
result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
total = result.sum()
pct = (result / total * 100).round(2)
summary = pd.DataFrame({'revenue': result.round(2), 'pct_of_total': pct})
print(summary.to_string())
```

**Time-series trend (monthly):**
```python
import pandas as pd
df = pd.read_csv('<csv_path>')
df['date'] = pd.to_datetime(df['date'])
monthly = df.set_index('date').resample('ME')['revenue'].sum().round(2)
print(monthly.to_string())
```

**Top-N ranking:**
```python
import pandas as pd
df = pd.read_csv('<csv_path>')
top = df.groupby('product').agg(
    total_revenue=('revenue', 'sum'),
    total_quantity=('quantity', 'sum'),
    order_count=('revenue', 'count')
).sort_values('total_revenue', ascending=False).round(2)
print(top.to_string())
```

### 4. Generate Charts

For trends and comparisons, save a chart:

**Bar chart for categorical comparison:**
```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('<csv_path>')
data = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
data.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Total Revenue by Region')
ax.set_ylabel('Revenue ($)')
ax.set_xlabel('Region')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/revenue_by_region.png', dpi=150)
print("Chart saved to output/revenue_by_region.png")
```

**Line chart for time-series:**
```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('<csv_path>')
df['date'] = pd.to_datetime(df['date'])
monthly = df.set_index('date').resample('ME')['revenue'].sum()
fig, ax = plt.subplots(figsize=(10, 5))
monthly.plot(kind='line', marker='o', ax=ax, color='steelblue')
ax.set_title('Monthly Revenue Trend')
ax.set_ylabel('Revenue ($)')
ax.set_xlabel('Month')
plt.tight_layout()
plt.savefig('output/monthly_revenue_trend.png', dpi=150)
print("Chart saved to output/monthly_revenue_trend.png")
```

## Output Expectations

- Present numeric results in small tables with clear headers
- Always include both raw values and percentages for comparisons
- Reference any generated charts by their file path
- Highlight the key takeaway from each computation
