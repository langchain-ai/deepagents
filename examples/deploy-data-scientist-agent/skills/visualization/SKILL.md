---
name: visualization
description: Create clear charts from one or more validated datasets, choosing chart types and aggregation levels appropriate to the data.
---

# Visualization

Use this skill when creating plots or explaining visual trends.

## Chart Selection

- Use line charts for time trends.
- Use bar charts for ranked categories or segment comparisons.
- Use scatter plots for relationships between two numeric variables.
- Use heatmaps sparingly for correlation or matrix-style summaries.
- For multiple files, choose charts only after confirming whether the files can be joined, concatenated, or compared side by side.

## Standards

- Always include a clear title, axis labels, and units when known.
- Sort categorical bars by value unless chronology or business order matters.
- Avoid overcrowded charts; aggregate or facet when there are many categories.
- Save each chart as its own `.png` file with a descriptive name.
- Save the Python script that generated each chart as its own `.py` file with a matching descriptive name.
- Display each generated chart separately after saving it; do not rely only on mentioning the path.
- Mention both the chart file path and the script file path in the final response.
- Include the source file name or combined dataset name in chart titles, captions, or report text when multiple files are involved.
- Do not overwrite earlier visuals or scripts during the same analysis; create a new descriptive filename if the chart changes.

## Validation

Before charting, confirm the aggregation level. For example, do not chart raw rows if the question asks for monthly revenue by region; group by month and region first. When charting multiple files, confirm that compared values use compatible definitions and units.
