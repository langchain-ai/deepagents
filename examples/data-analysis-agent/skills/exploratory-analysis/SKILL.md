---
name: exploratory-analysis
description: Use this skill for first-pass exploration of a dataset — understanding its shape, column types, missing values, distributions, and correlations
---

# Exploratory Analysis Skill

This skill provides a structured workflow for initial dataset exploration before deeper analysis.

## When to Use This Skill

Use this skill when asked to:
- Summarize or describe a dataset
- Explore what data is available
- Understand the structure of a CSV file
- Perform an initial data quality check
- Answer general questions like "What does this data look like?"

## Workflow

### 1. Dataset Overview

Run a script via `execute` to get basic shape and column info:

```python
import pandas as pd
df = pd.read_csv('<csv_path>')
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
```

Record the row count, column count, and column names.

### 2. Missing Value Audit

Check for nulls and flag any column exceeding 5% missing:

```python
import pandas as pd
df = pd.read_csv('<csv_path>')
total = len(df)
nulls = df.isnull().sum()
pct = (nulls / total * 100).round(2)
report = pd.DataFrame({'nulls': nulls, 'pct_missing': pct})
print(report[report['nulls'] > 0].to_string() if report['nulls'].any() else "No missing values")
```

### 3. Numeric Distributions

Compute summary statistics for all numeric columns:

```python
import pandas as pd
df = pd.read_csv('<csv_path>')
print(df.describe().round(2).to_string())
```

### 4. Categorical Breakdown

For each non-numeric column, show value counts:

```python
import pandas as pd
df = pd.read_csv('<csv_path>')
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col} — {df[col].nunique()} unique values:")
    print(df[col].value_counts().head(10).to_string())
```

### 5. Correlation Matrix (if applicable)

If there are 2+ numeric columns, compute pairwise correlations and save a heatmap:

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('<csv_path>')
numeric = df.select_dtypes(include='number')
if numeric.shape[1] >= 2:
    corr = numeric.corr().round(2)
    print(corr.to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('output/correlation_matrix.png', dpi=150)
    print("\nChart saved to output/correlation_matrix.png")
```

## Output Expectations

After running all steps, compile your observations into findings:
- Dataset shape and column summary
- Any data quality issues (missing values, unexpected types)
- Key distributions and notable patterns
- Correlations worth investigating further
