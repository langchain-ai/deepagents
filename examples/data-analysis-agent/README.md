# Data Analysis Agent

A CSV data analysis agent that explores datasets, runs Python/pandas scripts, and produces structured reports with charts.

**This example demonstrates:**
- **`LocalShellBackend`** – runs Python scripts via the `execute` tool (pandas, matplotlib)
- **Structured output** (`ToolStrategy`) – returns a typed `DataReport` Pydantic model
- **Memory** (`AGENTS.md`) – analytical standards (rounding, chart conventions, reporting rules)
- **Skills** (`skills/*/SKILL.md`) – workflows for exploratory analysis, statistical summaries, and anomaly detection

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

1. Clone the deepagents repository and navigate to this example:

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/examples/data-analysis-agent
```

2. Create a virtual environment and install dependencies:

```bash
# Using uv (recommended)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### Usage

```bash
# Summarize the bundled sample dataset
python analyze.py data/sample_sales.csv

# Ask a specific question
python analyze.py data/sample_sales.csv "Which region has the highest revenue?"
python analyze.py data/sample_sales.csv "What are the monthly revenue trends?"
python analyze.py data/sample_sales.csv "Find anomalies in daily revenue"
```

You can also point it at your own CSV files:
```bash
python analyze.py ~/Downloads/my_data.csv "Summarize this dataset"
```

## How It Works

The agent is configured by files on disk:

```
data-analysis-agent/
├── AGENTS.md                    # Analysis standards & reporting rules
├── skills/
│   ├── exploratory-analysis/
│   │   └── SKILL.md             # Dataset overview workflow
│   ├── statistical-summary/
│   │   └── SKILL.md             # Aggregation & trend workflow
│   └── anomaly-detection/
│       └── SKILL.md             # Outlier detection workflow
├── data/
│   └── sample_sales.csv         # Bundled demo dataset
└── analyze.py                   # Wires it together
```

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Rounding rules, chart conventions, reporting style | Always (system prompt) |
| `skills/*/SKILL.md` | Analysis-specific workflows with example scripts | On demand |

**What's in the skills?** Each skill teaches the agent a specific analysis workflow:
- **Exploratory analysis:** Dataset shape, column types, missing values, distributions, correlation matrix
- **Statistical summary:** Group-by aggregations, top-N rankings, time-series trends, pivot tables
- **Anomaly detection:** IQR outliers, z-score method, time-series deviation with rolling averages

## Architecture

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    backend=LocalShellBackend(     # ← enables the execute tool
        root_dir=EXAMPLE_DIR,
        virtual_mode=False,
        inherit_env=True,
    ),
    response_format=ToolStrategy(schema=DataReport),  # ← structured output
)
```

- **`LocalShellBackend`** implements `SandboxBackendProtocol`, which automatically enables the `execute` tool. The agent uses it to run self-contained Python scripts with pandas and matplotlib.
- **`ToolStrategy(schema=DataReport)`** tells the agent to return a typed Pydantic model instead of free text. The result is available at `response["structured_response"]`.

**Flow:**
1. Agent receives a CSV path and a question
2. Selects the relevant skill (exploratory, statistical, or anomaly detection)
3. Plans multi-step analyses with `write_todos`
4. Runs Python/pandas scripts via `execute` to compute statistics and generate charts
5. Returns a structured `DataReport` with findings and recommendations

## Output

The agent produces two types of output:

**Terminal:** A formatted report with dataset overview, color-coded findings (info/warning/critical), and recommendations.

**Files:** Charts are saved to `output/`:
```
output/
├── correlation_matrix.png
├── revenue_by_region.png
├── monthly_revenue_trend.png
└── anomaly_detection.png
```

## Customizing

**Change analysis standards:** Edit `AGENTS.md` to modify rounding precision, chart style, or reporting conventions.

**Add an analysis workflow:** Create `skills/<name>/SKILL.md` with YAML frontmatter:
```yaml
---
name: forecasting
description: Use this skill for time-series forecasting and projections
---
# Forecasting Skill
...
```

**Use your own data:** Point the agent at any CSV:
```bash
uv run python analyze.py /path/to/your/data.csv "What patterns do you see?"
```

## Security Note

This agent uses `LocalShellBackend`, which executes shell commands directly on your machine without sandboxing. It runs Python scripts to analyze data. Review the `AGENTS.md` and skills to understand what commands it may run, and avoid pointing it at directories with sensitive data.

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` – for the main agent (Claude Sonnet 4.5)
