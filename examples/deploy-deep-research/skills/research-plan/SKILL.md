---
name: research-plan
description: >-
  Create a structured research plan for a topic.
  Use when: the user provides a research question or topic,
  or when you need to break down a complex question into
  sub-questions before delegating to the researcher subagent.
---

# Research Planning

When given a research topic, decompose it into focused sub-questions.

## Process

1. **Restate the question** — make it precise and unambiguous
2. **Identify dimensions** — what aspects need investigation? (technical, historical, comparative, practical)
3. **Write sub-questions** — each should be answerable independently with web research
4. **Order by dependency** — if sub-question B depends on A's answer, A goes first
5. **Estimate depth** — mark each as "quick lookup" vs "deep dive"

## Output format

```markdown
## Research Plan: [restated question]

### Sub-questions

1. [question] — [quick lookup | deep dive]
2. [question] — [quick lookup | deep dive]
3. [question] — [quick lookup | deep dive]
```

Keep it to 3-5 sub-questions. Fewer is better — each one becomes a researcher task.
