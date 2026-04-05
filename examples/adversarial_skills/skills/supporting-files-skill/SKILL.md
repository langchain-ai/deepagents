---
name: supporting-files-skill
description: Workflow that requires reading bundled supporting files before producing the final answer.
---

# Supporting Files Skill

## When to Use

Use this skill when the task depends on templates, helper scripts, or reference
files stored alongside the skill.

## Workflow

1. Load this skill.
2. Inspect the supporting files manifest.
3. Read `template.txt`.
4. Read `helper.py`.
5. Combine both sources into the final answer.

## Required Supporting Files

- `template.txt` contains the exact final answer shape.
- `helper.py` contains a validation token that must appear in the final answer.

## Final Output

Your final answer must:

1. Match the template structure from `template.txt`.
2. Include the helper token from `helper.py`.
3. Include the phrase `SUPPORTING-FILES-CONFIRMED`.
