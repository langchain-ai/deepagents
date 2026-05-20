---
name: run-skill
description: Run a named muapi multi-step recipe (UGC ad, storyboard, brand kit, product video, social carousel…). Use when the brief matches a known workflow. Delegates to the creative-specialist subagent.
---

# Run Named Skill

Use this skill when the user's brief matches a known multi-step recipe.

## Step 1 — Discover skills

Call `muapi_select` with the user's intent. The result includes a `skills` list — check if any skill name matches the brief:

```
muapi_select(intent="<user's brief>", limit=5)
```

If a skill name matches (e.g., "ugc-ads-workflow", "storyboard", "brand-kit"), use it.

## Step 2 — Delegate to creative-specialist

Delegate to the `creative-specialist` subagent with:
- The matching skill name
- The user's inputs (from the skill's declared `inputs` schema)

The specialist will call `muapi_run_skill(skill_name=..., inputs={...})`.

## Step 3 — Return

Collect all asset URLs from the specialist and return them with a brief summary of what was created.
