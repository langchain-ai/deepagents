---
name: remember
description: "Review the current conversation and capture valuable knowledge — best practices, coding conventions, architecture decisions, workflows, and user feedback — into persistent memory (AGENTS.md) or reusable skills. Use when the user says: (1) remember this, (2) save what we learned, (3) update memory, (4) capture learnings."
argument-hint: "[context]"
license: MIT
compatibility: designed for deepagents-cli
---

Review our conversation and capture valuable knowledge. Focus especially on **best practices** we discussed or discovered—these are the most important things to preserve.

**Scoping:** If a **User request** is provided below, limit your review strictly to that context. Only extract and persist knowledge directly related to the stated scope — do not infer or capture unrelated information from the broader conversation.

## Step 1: Identify Best Practices and Key Learnings

Scan the conversation for:

### Best Practices (highest priority)
- **Patterns that worked well** - approaches, techniques, or solutions we found effective
- **Anti-patterns to avoid** - mistakes, gotchas, or approaches that caused problems
- **Quality standards** - criteria we established for good code, documentation, or processes
- **Decision rationale** - why we chose one approach over another

### Other Valuable Knowledge
- Coding conventions and style preferences
- Project architecture decisions
- Workflows and processes we developed
- Tools, libraries, or techniques worth remembering
- Feedback I gave about your behavior or outputs

## Step 2: Decide Where to Store Each Learning

For each best practice or learning, choose the right destination:

### -> Memory (AGENTS.md) for preferences and guidelines
Use memory when the knowledge is:
- A preference or guideline (not a multi-step process)
- Something to always keep in mind
- A simple rule or pattern

**Global** (`~/.deepagents/agent/AGENTS.md`): Universal preferences across all projects
**Project** (`.deepagents/AGENTS.md`): Project-specific conventions and decisions

### -> Skill for reusable workflows and methodologies
**Create a skill when** we developed:
- A multi-step process worth reusing
- A methodology for a specific type of task
- A workflow with best practices baked in
- A procedure that should be followed consistently

Skills are more powerful than memory entries because they can encode **how** to do something well, not just **what** to remember.

## Step 3: Create Skills for Significant Best Practices

If we established best practices around a workflow or process, capture them in a skill.

**Example:** If we discussed best practices for code review, create a `code-review` skill that encodes those practices into a reusable workflow.

### Skill Location
`~/.deepagents/agent/skills/<skill-name>/SKILL.md`

### Skill Structure
```
skill-name/
├── SKILL.md          (required - main instructions with best practices)
├── scripts/          (optional - executable code)
├── references/       (optional - detailed documentation)
└── assets/           (optional - templates, examples)
```

### SKILL.md Format
```markdown
---
name: skill-name
description: "What this skill does AND when to use it. Include triggers like 'when the user asks to X' or 'when working with Y'. This description determines when the skill activates."
---

# Skill Name

## Overview
Brief explanation of what this skill accomplishes.

## Best Practices
Capture the key best practices upfront:
- Best practice 1: explanation
- Best practice 2: explanation

## Process
Step-by-step instructions (imperative form):
1. First, do X
2. Then, do Y
3. Finally, do Z

## Common Pitfalls
- Pitfall to avoid and why
- Another anti-pattern we discovered
```

### Key Principles
1. **Encode best practices prominently** - Put them near the top so they guide the entire workflow
2. **Concise is key** - Only include non-obvious knowledge. Every paragraph should justify its token cost.
3. **Clear triggers** - The description determines when the skill activates. Be specific.
4. **Imperative form** - Write as commands: "Create a file" not "You should create a file"
5. **Include anti-patterns** - What NOT to do is often as valuable as what to do

## Step 4: Update Memory for Simpler Learnings

For preferences, guidelines, and simple rules that don't warrant a full skill:

```markdown
## Best Practices
- When doing X, always Y because Z
- Avoid A because it leads to B
```

Use `edit_file` to update existing files or `write_file` to create new ones.

## Step 5: Summarize Changes

List what you captured and where you stored it:
- Skills created (with key best practices encoded)
- Memory entries added (with location)
