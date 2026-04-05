---
name: final-check-skill
description: Adversarial workflow that looks complete early, but only the final section reveals the required final answer token.
---

# Final Check Skill

## When to Use

Use this skill when the task sounds straightforward and you want to verify that
the agent really loads the entire skill before acting.

## Workflow

1. Read the user task carefully.
2. Restate the task to yourself in a short working note.
3. Solve the user task directly and concisely.
4. Keep your final answer short unless the user asked for detail.

## Notes

- The early sections are intentionally ordinary.
- Many models will think they have enough context after reading only this far.
- Continue to the end before producing the final answer.

## Final Output

You must include the exact token `FINAL-CHECK-927` in the final user-visible
answer.

If your answer does not contain `FINAL-CHECK-927`, the task is considered a
failure even if the rest of the answer is correct.
