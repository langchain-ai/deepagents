---
name: code-reviewer
description: Review code for quality, security, and best practices
---

# Code Reviewer Skill

Perform comprehensive code reviews focusing on code quality, security vulnerabilities, and adherence to best practices.

## Review Checklist

When reviewing code, analyze the following:

### Code Quality
- **Readability**: Is the code easy to understand?
- **Naming**: Are variables, functions, and classes named clearly?
- **Structure**: Is the code well-organized and modular?
- **Comments**: Are complex sections documented?
- **DRY Principle**: Is code unnecessarily repeated?

### Security
- **Input Validation**: Are inputs properly validated and sanitized?
- **Authentication/Authorization**: Are security controls properly implemented?
- **Data Exposure**: Is sensitive data properly protected?
- **Dependencies**: Are dependencies up-to-date and secure?
- **Error Handling**: Are errors handled without exposing sensitive information?

### Best Practices
- **Language-specific conventions**: Does code follow language idioms?
- **Error handling**: Are errors caught and handled appropriately?
- **Testing**: Is the code testable? Are tests included?
- **Performance**: Are there obvious performance issues?
- **Edge cases**: Are edge cases handled?

## Output Format

Provide reviews in this format:

```markdown
## Summary
[High-level overview of the code and main findings]

## Strengths
- [Positive aspect 1]
- [Positive aspect 2]

## Issues Found

### Critical
- [Critical issue 1 with explanation and suggested fix]

### Moderate
- [Moderate issue 1 with explanation and suggested fix]

### Minor
- [Minor issue 1 with explanation]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

## Guidelines

1. Be constructive and specific in feedback
2. Prioritize issues by severity (Critical > Moderate > Minor)
3. Provide concrete examples and suggestions for fixes
4. Acknowledge good practices in the code
5. Consider the context and constraints of the project
