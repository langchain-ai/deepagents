# Code Review Agent

An AI code review agent that analyzes code for bugs, security vulnerabilities, performance issues, and suggests improvements.

**This example demonstrates:**
- **Memory** (`AGENTS.md`) — review philosophy and structured output format
- **Skills** (`skills/*/SKILL.md`) — specialized review workflows loaded on demand
- **Subagents** (`subagents.yaml`) — dedicated security analyzer for deep analysis
- **LocalShellBackend** — filesystem access for code exploration

## Quick Start

```bash
# Set API key (get one at https://ai.google.dev/gemini-api/docs)
export GOOGLE_API_KEY="..."

# Run (uv automatically installs dependencies on first run)
cd examples/code-review-agent
uv run python agent.py /path/to/code
```

## Usage

```bash
# Review a directory
uv run python agent.py ./src/

# Review a single file
uv run python agent.py ./src/main.py

# Security-focused review
uv run python agent.py ./src/ --focus security

# Performance-focused review
uv run python agent.py ./src/ --focus performance

# Custom output file
uv run python agent.py ./src/ -o my-review.md
```

## How It Works

The agent is configured by files on disk:

```
code-review-agent/
├── AGENTS.md                    # Review philosophy & output format
├── subagents.yaml               # Security analyzer subagent
├── skills/
│   ├── general-review/
│   │   └── SKILL.md             # Bug hunting, code quality workflow
│   ├── security-review/
│   │   └── SKILL.md             # OWASP-style vulnerability analysis
│   └── performance-review/
│       └── SKILL.md             # Efficiency & scalability review
└── agent.py                     # Wires it together
```

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Review philosophy, output format, tool guidance | Always (system prompt) |
| `subagents.yaml` | Security analyzer subagent | Always (defines `task` tool) |
| `skills/general-review/` | Bug hunting, error handling, code quality | On demand |
| `skills/security-review/` | Injection, auth, secrets, crypto analysis | On demand |
| `skills/performance-review/` | N+1 queries, complexity, memory issues | On demand |

## Architecture

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],                        # ← Loaded into system prompt
    skills=["./skills/"],                          # ← Loaded on demand
    subagents=load_subagents("./subagents.yaml"),  # ← Security analyzer
    backend=LocalShellBackend(root_dir=target, inherit_env=True),
)
```

The agent uses `LocalShellBackend` which provides filesystem tools (`read_file`, `write_file`, `glob`, `grep`, `ls`) and shell execution via the `execute` tool.

**Flow:**
1. Agent receives task → selects relevant skill (general, security, or performance)
2. Plans the review with `write_todos`
3. Explores code using `read_file`, `grep`, `glob`, `ls`
4. Optionally delegates to `security-analyzer` subagent for deep security analysis
5. Writes structured review report to disk

## Output

The agent produces a markdown report (default: `review-report.md`):

```markdown
# Code Review Report

## Summary
Good code quality overall with two issues that should be addressed before merge.

## Critical Issues
### 1. SQL Injection in user query handler
- **File**: src/db.py:45
- **Issue**: User input concatenated directly into SQL query
- **Fix**: Use parameterized queries

## Warnings
### 1. Missing error handling for API calls
- **File**: src/api.py:23
- ...

## Suggestions
### 1. Consider extracting duplicated validation logic
- ...

## Files Reviewed
- src/main.py — Entry point, clean structure
- src/db.py — Database layer, has injection issue
- src/api.py — API client, missing error handling

## Positive Observations
- Consistent naming conventions throughout
- Good use of type hints
```

If security analysis is delegated, a separate `security-analysis.md` is also produced.

## Customizing

**Change review style:** Edit `AGENTS.md` to modify the output format or review priorities.

**Add a language-specific skill:** Create `skills/<name>/SKILL.md` with YAML frontmatter:
```yaml
---
name: python-review
description: Use this skill for Python-specific code review (type hints, async patterns, packaging)
---
# Python Review Skill
...
```

**Change the security subagent:** Edit `subagents.yaml` to modify the security analyzer's focus areas or system prompt.

## Security Note

This agent uses `LocalShellBackend` which can execute shell commands and read/write files. It runs with your user permissions. Only review code you trust, in a safe environment.

## Requirements

- Python 3.11+
- `GOOGLE_API_KEY` — For the main agent (Gemini)
