---
name: anomaly-detection
description: Use this skill to detect outliers, anomalies, and unusual patterns in numeric or time-series data
---

# Anomaly Detection Skill

This skill provides a workflow for identifying outliers and anomalous data points using statistical methods.

## When to Use This Skill

Use this skill when asked to:
- Find outliers or unusual values
- Detect anomalies in time-series data
- Identify suspicious transactions or records
- Check for data entry errors
- Answer questions like "Are there any anomalies?" or "What looks unusual?"

## Workflow

### 1. Identify Target Columns

Determine which numeric columns to check for anomalies. Prioritize:
- Revenue/monetary columns (unexpected spikes or drops)
- Quantity columns (unusually large or zero values)
- Time-series metrics (deviations from trend)

### 2. IQR Method (Interquartile Range)

The IQR method flags values outside 1.5x the interquartile range. Good for general outlier detection:

```python
import pandas as pd
df = pd.read_csv('<csv_path>')

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        print(f"\n{col}: {len(outliers)} outliers (range: {lower:.2f} to {upper:.2f})")
        print(outliers[[col]].head(10).to_string())
    else:
        print(f"\n{col}: no outliers detected")
```

### 3. Z-Score Method

For normally distributed data, flag values with |z-score| > 3:

```python
import pandas as pd
import numpy as np
df = pd.read_csv('<csv_path>')

for col in df.select_dtypes(include='number').columns:
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        continue
    df[f'{col}_zscore'] = ((df[col] - mean) / std).round(2)
    anomalies = df[df[f'{col}_zscore'].abs() > 3]
    if len(anomalies) > 0:
        print(f"\n{col}: {len(anomalies)} anomalies (z-score > 3)")
        print(anomalies[['date', col, f'{col}_zscore']].head(10).to_string() if 'date' in df.columns else anomalies[[col, f'{col}_zscore']].head(10).to_string())
```

### 4. Time-Series Deviation (if date column exists)

Compare each period's value against a rolling average to spot sudden changes:

```python
import pandas as pd
import numpy as np
df = pd.read_csv('<csv_path>')
df['date'] = pd.to_datetime(df['date'])

daily = df.set_index('date').resample('D')['revenue'].sum()
rolling_mean = daily.rolling(window=7, min_periods=1).mean()
rolling_std = daily.rolling(window=7, min_periods=1).std().fillna(0)
deviation = ((daily - rolling_mean) / rolling_std.replace(0, np.nan)).dropna()
anomalies = deviation[deviation.abs() > 2]

if len(anomalies) > 0:
    print(f"Time-series anomalies (>2 std from 7-day rolling mean): {len(anomalies)}")
    for date, score in anomalies.items():
        actual = daily.loc[date]
        expected = rolling_mean.loc[date]
        print(f"  {date.date()}: actual={actual:.2f}, expected={expected:.2f}, deviation={score:.2f}x std")
else:
    print("No time-series anomalies detected")
```

### 5. Visualize Anomalies

Save a chart highlighting outlier points:

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('<csv_path>')
df['date'] = pd.to_datetime(df['date'])
daily = df.set_index('date').resample('D')['revenue'].sum()
rolling_mean = daily.rolling(window=7, min_periods=1).mean()
rolling_std = daily.rolling(window=7, min_periods=1).std().fillna(0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily.index, daily.values, 'o-', markersize=3, label='Daily Revenue', alpha=0.7)
ax.plot(rolling_mean.index, rolling_mean.values, '-', color='orange', linewidth=2, label='7-day Rolling Mean')
ax.fill_between(rolling_mean.index,
                (rolling_mean - 2 * rolling_std).values,
                (rolling_mean + 2 * rolling_std).values,
                alpha=0.15, color='orange', label='±2 Std Dev')

deviation = ((daily - rolling_mean) / rolling_std.replace(0, np.nan)).dropna()
anomaly_dates = deviation[deviation.abs() > 2].index
if len(anomaly_dates) > 0:
    ax.scatter(anomaly_dates, daily.loc[anomaly_dates].values,
               color='red', s=60, zorder=5, label='Anomalies')

ax.set_title('Daily Revenue with Anomaly Detection')
ax.set_ylabel('Revenue ($)')
ax.set_xlabel('Date')
ax.legend()
plt.tight_layout()
plt.savefig('output/anomaly_detection.png', dpi=150)
print("Chart saved to output/anomaly_detection.png")
```

## Output Expectations

- List each anomaly with its specific value and why it was flagged (which method, threshold)
- Provide context: what is the normal range and how far the anomaly deviates
- Classify severity: **critical** for values >3 std deviations, **warning** for 2-3 std deviations
- Reference any generated anomaly charts
- Recommend whether anomalies look like data errors or genuine unusual events
