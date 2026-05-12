---
name: code-review
description: Perform a structured code review of changes, checking for correctness, style, tests, and potential issues.
---

# Code Review Skill

Use this skill after implementing changes to validate your work before delivering.

## Review Checklist

### 1. Correctness
- [ ] Changes solve the original issue/task
- [ ] No unintended side effects on existing functionality
- [ ] Edge cases are handled
- [ ] Error handling is appropriate (not excessive)

### 2. Code Quality
- [ ] Code matches existing style and patterns
- [ ] No unnecessary complexity or abstraction
- [ ] Variable and function names are clear
- [ ] No dead code, commented-out code, or TODOs left behind

### 3. Tests
- [ ] New functionality has test coverage
- [ ] Existing tests still pass
- [ ] Tests cover both happy path and error cases
- [ ] Tests are not brittle (don't test implementation details)

### 4. Safety
- [ ] No hardcoded secrets or credentials
- [ ] User input is validated at boundaries
- [ ] No SQL injection, XSS, or command injection vectors
- [ ] File operations use safe paths

## Process

1. Read each modified file end-to-end (not just the diff)
2. Run the test suite: `execute("python -m pytest -v")`
3. Run linters if available: `execute("ruff check .")`
4. Run the bundled lint check: `execute("python /skills/code-review/lint_check.py .")`
5. Check against each item in the review checklist
6. If any issues found, fix them and re-review
7. When everything passes, the review is complete

## Helper Scripts

- **`/skills/code-review/lint_check.py`** — Scans Python files for missing
  docstrings, long functions (>50 lines), and bare `except:` clauses. Run it
  via `execute("python /skills/code-review/lint_check.py [path ...]")`.
