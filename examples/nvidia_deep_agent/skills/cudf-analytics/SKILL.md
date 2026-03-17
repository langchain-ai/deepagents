---
name: cudf-analytics
description: Use for GPU-accelerated data analysis on datasets, CSVs, or tabular data using NVIDIA cuDF. Triggers when tasks involve groupby aggregations, statistical summaries, anomaly detection, or large-scale data profiling.
---

# cuDF Analytics Skill

GPU-accelerated data analysis using NVIDIA RAPIDS cuDF. cuDF provides a pandas-like API that runs on NVIDIA GPUs, enabling massive speedups on large datasets.

## When to Use This Skill

Use this skill when:
- Analyzing CSV files, datasets, or tabular data
- Computing statistical summaries (mean, median, std, quartiles)
- Performing groupby aggregations
- Detecting anomalies or outliers in data
- Profiling datasets with millions of rows
- Computing correlation matrices

## Initialization (REQUIRED)

Always start every script with this boilerplate. It tests actual GPU operations, not just import.

```python
import pandas as pd

try:
    import cudf
    # Smoke-test: verify GPU compute AND host transfer both work
    _test = cudf.Series([1, 2, 3])
    assert _test.sum() == 6
    assert _test.to_pandas().tolist() == [1, 2, 3]
    GPU = True
except Exception as e:
    print(f"[GPU] cudf unavailable, falling back to pandas: {e}")
    GPU = False

def read_csv(path):
    return cudf.read_csv(path) if GPU else pd.read_csv(path)

def to_pd(df):
    """Convert cuDF DataFrame/Series to pandas. Use this instead of .to_pandas() directly."""
    if not GPU:
        return df
    try:
        return df.to_pandas()
    except Exception as e:
        print(f"[GPU] .to_pandas() failed, using Arrow fallback: {e}")
        return df.to_arrow().to_pandas()
```

## Quick Reference

cuDF mirrors the pandas API. Common operations:

### Read Data
```python
df = read_csv("data.csv")
```

### Statistical Summary
```python
# Use to_pd() when you need pandas output
summary = to_pd(df[["value", "score"]].describe())

# Scalar values work directly with float()
mean_val = float(df["value"].mean())
q1 = float(df["value"].quantile(0.25))

# Correlation
corr = float(df["value"].corr(df["score"]))
```

### Groupby Aggregation
```python
result = df.groupby("category").agg({
    "revenue": ["sum", "mean", "count"],
    "quantity": ["sum", "mean"],
})
result_pd = to_pd(result)
```

### Anomaly Detection (IQR Method)
```python
col = "value"
Q1 = float(df[col].quantile(0.25))
Q3 = float(df[col].quantile(0.75))
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = to_pd(df[(df[col] < lower) | (df[col] > upper)])
```

### Anomaly Detection (Z-Score Method)
```python
mean = float(df[col].mean())
std = float(df[col].std())
df["z_score"] = (df[col] - mean) / std
anomalies = to_pd(df[df["z_score"].abs() > 3])
```

### Filtering and Selection
```python
# Filter rows
filtered = df[df["status"] == "active"]

# Select columns
subset = df[["name", "revenue", "date"]]

# Sort
sorted_df = df.sort_values("revenue", ascending=False)

# Convert to pandas for final output / iteration
result_pd = to_pd(sorted_df)
```

## Data Type Requirements

cuDF requires explicit type specification for optimal performance:
- Use `float32` or `float64` for numeric data
- Use `int32` or `int64` for integer data
- String columns use cuDF's string dtype automatically

## Output Guidelines

When reporting analysis results:
- Include dataset dimensions (rows x columns)
- Show key statistics in formatted tables
- Highlight notable patterns, trends, or anomalies
- Provide both summary statistics and specific examples
- Note any data quality issues (missing values, outliers)
