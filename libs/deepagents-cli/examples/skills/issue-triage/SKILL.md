---
name: issue-triage
description: Triage GitHub issues for a repository - categorize issues, find patterns, identify code hotspots, and generate an actionable report. Use when asked to analyze, triage, or understand issues in a repo.
---

# Issue Triage Skill

Analyze GitHub issues to find patterns, categorize problems, identify code hotspots, and generate an actionable triage report.

## Workflow

### Step 1: Fetch Issues

Use the fetch script to get issue data. Replace `OWNER/REPO` with the target repository:

```bash
python3 [SKILLS_DIR]/issue-triage/scripts/fetch_issues.py OWNER/REPO --days 90 --limit 200
```

Arguments:
- `OWNER/REPO` (required): Repository in format `owner/repo`
- `--days N`: Issues from last N days (default: 90)
- `--limit N`: Max issues to fetch (default: 200)
- `--state`: open, closed, or all (default: open)

The script outputs JSON to stdout. Save it:
```bash
python3 [SKILLS_DIR]/issue-triage/scripts/fetch_issues.py OWNER/REPO --days 90 > issues.json
```

### Step 2: Analyze the Issues

Read the JSON and analyze:

1. **Categorize each issue** by type:
   - Bug: something broken, error, crash, regression
   - Feature: new capability, enhancement, improvement
   - Question: how to, help needed, support
   - Docs: documentation issue, missing docs
   - Chore: maintenance, refactor, tech debt

2. **Find patterns** - look for:
   - Similar issues (same error, same component)
   - Duplicate reports of the same underlying problem
   - Clusters around specific features or code areas
   - Recurring themes in titles/bodies

3. **Identify code hotspots**:
   - Extract file paths, function names, error messages from issue bodies
   - Note which files are mentioned across multiple issues
   - For high-signal issues, read the referenced code to understand context

4. **Assess severity** based on:
   - Number of reactions (thumbs up = user demand)
   - Number of comments (engagement = importance)
   - How many issues stem from same root cause

### Step 3: Generate the Report

Create `TRIAGE.md` in the current directory with this structure:

```markdown
# Issue Triage: owner/repo
> X open issues | Last N days | Generated YYYY-MM-DD

## Summary

| Pattern | Issues | Top Issue | Signal |
|---------|--------|-----------|--------|
| Pattern Name | N | [#123](url) - brief desc | X reactions |
| Another Pattern | N | [#456](url) - brief desc | Y comments |
...

**Health:** One-line assessment (e.g., "Needs attention - 64% unlabeled, patterns around X and Y")

## Patterns Detail

### Pattern Name (N issues)
Brief description of the pattern.
- [#123](url): "Issue title"
- [#456](url): "Issue title"

> "Key quote from issue that illustrates the problem" - #123

**Code hotspot**: `path/to/file.ts` (if applicable)
**Root cause hypothesis**: Your assessment (if you investigated code)

[Repeat for each pattern discovered]

## Code Hotspots
| File | Issues | Notes |
|------|--------|-------|
| path/to/file.ts | #1, #2, #3 | Brief note |

## Recommended Actions
1. **HIGH**: Action - reason (issues affected)
2. **MEDIUM**: Action - reason
3. **LOW**: Action - reason
```

The summary table is the most important part - it should give a quick overview of what patterns exist and where to look first.

### Step 4: Code Investigation (Optional)

If the user wants deeper analysis on specific issues:
1. Read the files mentioned in those issues
2. Understand the code context
3. Assess: is this a real bug? how complex to fix?
4. Update the report with your findings

## Tips

- Quote directly from issues to show evidence for patterns
- Link every issue reference: `[#123](https://github.com/owner/repo/issues/123)`
- Focus on actionable insights, not just listing issues
- If you find duplicates, note which issue is the "primary" one
- For large repos, focus on most recent and most reacted issues
