---
name: planning
description: Break down a coding task into a structured implementation plan with clear steps, file identification, and risk assessment.
---

# Planning Skill

Use this skill when starting a new coding task to create a thorough implementation plan.

## Steps

### 1. Understand the Task
- Read the issue/task description completely
- Identify the expected outcome and acceptance criteria
- Note any constraints or requirements mentioned

### 2. Explore the Codebase
- Find the repository root and read the project structure
- Identify the tech stack (language, framework, test runner)
- Read README, CONTRIBUTING, or similar docs if they exist
- Find existing tests to understand testing patterns

### 3. Identify Relevant Files
- Use `grep` to find code related to the task
- Read the most relevant files (entry points, related modules)
- Identify which files need to be modified vs. created
- Check for existing patterns you should follow

### 4. Write the Plan
Use `write_todos` to create a structured plan:

```
write_todos([
    "1. <specific change in specific file>",
    "2. <next specific change>",
    "3. Write tests for <feature>",
    "4. Run test suite and fix failures",
    "5. Review all changes"
])
```

### 5. Assess Risks
- Are there breaking changes?
- Are there edge cases to handle?
- Does this affect other parts of the codebase?
- Flag anything uncertain for review

## Guidelines

- Plans should have 3-10 concrete steps
- Each step should be specific enough to execute without further planning
- Include test writing and test running as explicit steps
- End with a review/verification step
