---
name: data-visualization
description: Use for creating publication-quality charts and multi-panel analysis summaries. Triggers when tasks involve visualizing data, plotting results, creating charts, or producing visual reports from analysis output.
---

# Data Visualization Skill

Create publication-quality analytical charts using matplotlib and seaborn in a headless GPU sandbox. Charts are saved as PNG files to `/workspace/` for retrieval.

## When to Use This Skill

Use this skill when:
- Visualizing results from cuDF analysis or cuML models
- Creating charts (bar, line, scatter, heatmap, histogram, box plot)
- Building multi-panel analysis summaries
- The user asks for visual output, plots, graphs, or charts
- Presenting statistical findings with figures

## Initialization (REQUIRED)

MUST call `matplotlib.use('Agg')` BEFORE importing pyplot. This enables headless rendering.

```python
import matplotlib
matplotlib.use('Agg')  # Headless backend — MUST be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality defaults
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.constrained_layout.use': True,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
          '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']
```

## Saving Charts

Always save to `/workspace/` with these settings:

```python
plt.savefig('/workspace/chart_name.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

- `dpi=300` for print quality
- `bbox_inches='tight'` removes excess whitespace
- `facecolor='white'` ensures white background
- Always call `plt.close()` after saving to free memory

## Displaying Charts (REQUIRED)

After saving any chart, you MUST call `read_file` on it to display it inline in the conversation:

```
read_file("/workspace/chart_name.png")
```

Users cannot see charts unless you do this. Every chart you save MUST be followed by a `read_file` call.

## Quick Reference

### Bar Chart (from groupby results)

```python
# After: result = to_pd(df.groupby("category")["value"].mean())
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(result.index, result.values, color=COLORS[:len(result)],
              edgecolor='black', linewidth=0.8)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Mean Value', fontweight='bold')
ax.set_xlabel('Category', fontweight='bold')
ax.set_title('Average Value by Category', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.savefig('/workspace/bar_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Line Chart (trends over time)

```python
fig, ax = plt.subplots(figsize=(10, 5))

for i, col in enumerate(columns_to_plot):
    ax.plot(df["date"], df[col], label=col, color=COLORS[i], linewidth=2,
            marker='o', markersize=3, markevery=max(1, len(df)//20))

ax.set_ylabel('Values', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')
ax.set_title('Trends Over Time', fontweight='bold')
ax.legend(frameon=True, shadow=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.xticks(rotation=45, ha='right')

plt.savefig('/workspace/line_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Scatter Plot — Continuous Color (correlations)

```python
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df["x"], df["y"], c=df["value"], cmap='viridis',
                     s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
plt.colorbar(scatter, ax=ax, label='Value')

# Optional: trend line
z = np.polyfit(df["x"], df["y"], 1)
ax.plot(df["x"].sort_values(), np.poly1d(z)(df["x"].sort_values()),
        "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_title('Correlation Analysis', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')

plt.savefig('/workspace/scatter_correlation.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Scatter Plot — Categorical Color (clusters)

```python
fig, ax = plt.subplots(figsize=(8, 6))

for i, label in enumerate(sorted(df["cluster"].unique())):
    mask = df["cluster"] == label
    ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
               c=COLORS[i], label=f'Cluster {label}', s=40, alpha=0.7)

ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_title('Cluster Visualization', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')

plt.savefig('/workspace/scatter_clusters.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Heatmap (correlation matrix or confusion matrix)

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 7))

# corr_matrix = to_pd(df[numeric_cols].corr())
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation'}, ax=ax)

ax.set_title('Correlation Matrix', fontweight='bold')

plt.savefig('/workspace/heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Histogram with KDE

```python
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df["value"], bins=30, color=COLORS[0], alpha=0.7,
        edgecolor='black', linewidth=0.5, density=True, label='Distribution')

# Add KDE curve
from scipy.stats import gaussian_kde
kde = gaussian_kde(df["value"].dropna())
x_range = np.linspace(df["value"].min(), df["value"].max(), 200)
ax.plot(x_range, kde(x_range), color=COLORS[1], linewidth=2, label='KDE')

ax.set_xlabel('Value', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Value Distribution', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.savefig('/workspace/histogram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Box Plot (compare groups)

```python
fig, ax = plt.subplots(figsize=(8, 5))

groups = [df[df["group"] == g]["value"].values for g in group_names]
bp = ax.boxplot(groups, labels=group_names, patch_artist=True,
                widths=0.6, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(COLORS[i % len(COLORS)])
    patch.set_alpha(0.7)

ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Distribution by Group', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.savefig('/workspace/boxplot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Multi-Panel Analysis Summary

Use this to create a single image with multiple charts — the most effective way to present a complete analysis.

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Distribution
axes[0, 0].hist(df["value"], bins=30, color=COLORS[0], alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0, 0].set_title('Value Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

# Top-right: Scatter
axes[0, 1].scatter(df["x"], df["y"], c=COLORS[0], s=30, alpha=0.5)
axes[0, 1].set_title('X vs Y', fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# Bottom-left: Bar chart
group_means = df.groupby("category")["value"].mean()
axes[1, 0].bar(group_means.index, group_means.values, color=COLORS[:len(group_means)])
axes[1, 0].set_title('Mean by Category', fontweight='bold')
axes[1, 0].set_xlabel('Category')
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

# Bottom-right: Box plot
axes[1, 1].boxplot([df[df["category"] == c]["value"].values for c in categories],
                    labels=categories, patch_artist=True)
axes[1, 1].set_title('Distribution by Category', fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Analysis Summary', fontsize=16, fontweight='bold')

plt.savefig('/workspace/analysis_summary.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Feature Importance Chart (from cuML model)

```python
fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))

# importances = to_pd(model.feature_importances_)
sorted_idx = np.argsort(importances)
ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx],
        color=COLORS[0], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Feature Importances', fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.savefig('/workspace/feature_importance.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

### Confusion Matrix (from cuML classification)

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 6))

# cm = confusion_matrix(to_pd(y_test), to_pd(predictions))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=1, cbar_kws={'label': 'Count'}, ax=ax)

ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('Actual', fontweight='bold')
ax.set_title('Confusion Matrix', fontweight='bold')

plt.savefig('/workspace/confusion_matrix.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
# IMPORTANT: call read_file("/workspace/<chart>.png") to display inline
```

## Style Rules

- Use `COLORS` palette (colorblind-safe) — never rely on color alone to distinguish elements
- No pie charts (bar charts are always clearer)
- No 3D plots (distort data perception)
- Grid lines at `alpha=0.3, linestyle='--'` with `ax.set_axisbelow(True)`
- Bold axis labels and titles (`fontweight='bold'`)
- White background for all exports
- 1-4 charts per analysis is typical; use multi-panel for more

## Output Guidelines

- Save all charts to `/workspace/` as PNG
- Print file paths after saving so the agent can reference them
- For multi-panel summaries, use `figsize=(14, 10)` for 2×2 layouts
- Keep chart titles descriptive but concise
- Include units in axis labels when applicable
