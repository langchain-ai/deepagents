---
name: performance-review
description: For performance-focused code review targeting inefficient algorithms, N+1 queries, memory issues, unnecessary computation, and scalability problems
---

# Performance Review Skill

A structured workflow for identifying performance issues in code.

## When to Use This Skill

Use this skill when asked to:
- Review code for performance issues
- Optimize slow code paths
- Check database query efficiency
- Review code that handles large datasets or high traffic

## Workflow

### 1. Identify Hot Paths

Determine which code runs frequently or handles large data:
```
glob("**/routes/**")           # API endpoints
glob("**/handlers/**")         # Request handlers
glob("**/models/**")           # Database models
glob("**/workers/**")          # Background jobs
```

Read entry points and trace the call paths.

### 2. Scan for Anti-Patterns

**N+1 Queries (database calls in loops):**
```
grep("for.*in.*:\\n.*\\.query\\(|for.*in.*:\\n.*\\.find\\(|for.*in.*:\\n.*\\.get\\(")
grep("for.*in.*:\\n.*SELECT|for.*in.*:\\n.*execute")
```

**Inefficient Loops:**
```
grep("for.*in.*for.*in")       # Nested loops (potential O(n^2))
```

**Unnecessary Work:**
```
grep("SELECT \\*")             # Fetching all columns
grep("\\.fetchall\\(")          # Loading entire result set into memory
grep("import.*\\*")            # Wildcard imports (slower startup)
```

**String Concatenation in Loops:**
```
grep("\\+= ['\"]|\\+= f['\"]")  # String building without join()
```

**Missing Caching:**
```
grep("@cache|@lru_cache|@cached|memoize")   # Check if caching is used
```

**Synchronous I/O in Async Context:**
```
grep("requests\\.get|requests\\.post|urllib\\.request")  # Sync HTTP in async code
grep("open\\(.*\\)|read\\(\\)|write\\(\\)")               # Sync file I/O
```

### 3. Analyze Complexity

For each flagged section:
- What is the time complexity? (O(n), O(n^2), O(n*m)?)
- What is the expected data size?
- Is this on a hot path (called per request) or cold path (called once)?
- Would the issue matter at 10x or 100x scale?

### 4. Check Resource Management

Look for:
- **Unbounded growth** — Lists/dicts that grow without limits
- **Missing pagination** — API endpoints returning all records
- **Connection leaks** — Database/HTTP connections not closed
- **Large file loading** — Reading entire files into memory instead of streaming
- **Missing timeouts** — Network calls without timeout parameters

### 5. Language-Specific Patterns

**Python:**
- List comprehension building a huge list vs generator expression
- Mutable default arguments (`def f(items=[])`)
- Global state in multi-threaded context
- `time.sleep()` in async code

**JavaScript/TypeScript:**
- Synchronous operations blocking the event loop
- Missing `await` on promises (fire-and-forget)
- Excessive DOM manipulation without batching
- Missing debounce/throttle on event handlers
- Large bundle imports (`import _ from 'lodash'` vs `import get from 'lodash/get'`)

**SQL:**
- Missing indexes on columns used in WHERE/JOIN
- `SELECT *` instead of specific columns
- Subqueries that could be JOINs
- Missing LIMIT on unbounded queries
- N+1 patterns from ORM lazy loading

### 6. Write the Report

Categorize findings as:
- **Algorithmic** — Poor time/space complexity
- **I/O** — Unnecessary or inefficient I/O operations
- **Database** — Query inefficiency, N+1, missing indexes
- **Memory** — Unbounded growth, large allocations
- **Concurrency** — Blocking operations, missing async

Include estimated impact (low/medium/high) based on how often the code path runs.

## Quality Checklist

Before finishing:
- [ ] Hot paths and high-traffic code identified
- [ ] Common anti-patterns scanned with grep
- [ ] Findings include complexity analysis where relevant
- [ ] Impact is estimated based on code path frequency
- [ ] Recommendations are specific and actionable
- [ ] Report is saved to the specified output file
