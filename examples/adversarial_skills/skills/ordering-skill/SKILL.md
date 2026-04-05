---
name: ordering-skill
description: Adversarial workflow that forbids file writes or edits until validation is completed and explicitly recorded.
---

# Ordering Skill

## When to Use

Use this skill for tasks that create or modify files and need strict sequencing.

## Workflow

1. Understand the user request.
2. Gather any information needed to construct the result.
3. Prepare a draft mentally or in a non-file output.

## Validation

Before calling `write_file` or `edit_file`, you must perform an explicit
validation step.

Validation means:

1. State that validation is happening.
2. Check that the intended output satisfies the user request.
3. Confirm the validation with the exact phrase `VALIDATION-COMPLETE-314`.

## Hard Constraint

Do not call `write_file` or `edit_file` before validation is complete.

If validation fails, stop and explain why instead of modifying files.

## Final Output

If a file was modified successfully, mention that validation happened first.
