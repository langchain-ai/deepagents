---
name: generate-asset
description: Generate a single media asset — image, video, audio, or 3D. Use when the user wants one output from a clear prompt. Calls muapi_select to pick the best model, then muapi_generate.
---

# Generate Asset

Use this skill for single-asset generation tasks.

## Step 1 — Discover

Call `muapi_select` with the user's intent and kind:

```
muapi_select(intent="<user's prompt>", kind="<image|video|audio|3d>", tier="<best|balanced|fast>", limit=3)
```

Pick the top-ranked model from the result.

## Step 2 — Generate

Call `muapi_generate` with the chosen model:

```
muapi_generate(
    prompt="<refined prompt>",
    kind="<kind>",
    model="<name from muapi_select>",
    tier="<best|balanced|fast>",
)
```

For edits, enhancements, image-to-video, or lipsync — pass `input_asset_url` too.

## Step 3 — Return

Report the asset URL to the user. Include: model used, kind, and any relevant parameters.
