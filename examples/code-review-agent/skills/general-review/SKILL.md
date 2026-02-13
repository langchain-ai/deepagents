---
name: general-review
description: For comprehensive code review covering bugs, error handling, logic issues, naming, structure, and maintainability
---

# General Code Review Skill

A structured workflow for comprehensive code quality review.

## When to Use This Skill

Use this skill when asked to:
- Perform a general code review
- Review a pull request or set of changes
- Check code quality without a specific focus area
- Review code for bugs and correctness

## Workflow

### 1. Explore the Code

Start by understanding what you're reviewing:

**For a directory:**
```
ls()                           # See top-level structure
glob("**/*.py")                # Find all Python files (adjust extension)
```

**For a single file:**
```
read_file("path/to/file")
```

### 2. Plan the Review

For multi-file reviews, use `write_todos` to organize:
```
write_todos([
    {"id": "1", "title": "Review main entry point", "status": "todo"},
    {"id": "2", "title": "Check error handling patterns", "status": "todo"},
    {"id": "3", "title": "Scan for common red flags", "status": "todo"},
    {"id": "4", "title": "Write review report", "status": "todo"},
])
```

### 3. Scan for Red Flags

Use `grep` to search for common issues across the codebase:

```
grep("TODO|FIXME|HACK|XXX")           # Unfinished work
grep("eval\\(|exec\\(")               # Dangerous dynamic execution
grep("except:|except Exception")       # Overly broad exception handling
grep("password|secret|api_key")        # Potential hardcoded secrets
grep("print\\(")                       # Debug prints left in code
```

### 4. Read and Analyze Files

Read each file and check for:

**Correctness:**
- Logic errors and off-by-one mistakes
- Null/undefined/None handling
- Missing return statements
- Incorrect boolean logic
- Unreachable code after return/break/continue

**Error Handling:**
- Bare except clauses that swallow errors
- Missing error handling for I/O, network, parsing
- Error messages that leak internal details
- Resources not cleaned up on error (missing finally/context managers)

**Code Structure:**
- Functions doing too many things
- Deeply nested conditionals (>3 levels)
- Duplicated code blocks
- Unused imports, variables, or functions
- Inconsistent naming conventions

**Edge Cases:**
- Empty inputs (empty strings, empty lists, None)
- Boundary values (0, -1, MAX_INT)
- Concurrent access issues
- Race conditions in shared state

### 5. Write the Report

Save the review report using `write_file` following the format in AGENTS.md.

## Quality Checklist

Before finishing your review:
- [ ] All files in scope have been read or scanned
- [ ] Red flag grep patterns have been checked
- [ ] Each finding has a specific file and line reference
- [ ] Each finding explains WHY it's a problem
- [ ] Findings are categorized by severity (Critical/Warning/Suggestion)
- [ ] Positive observations are included
- [ ] Report is saved to the specified output file
