---
name: visual-build
description: "Verify-before-guess workflow for visual/front-end build and tweak tasks. Use when the user asks to build or adjust a webpage, page layout, visual, slide deck, or self-contained HTML with visuals, and for follow-up cosmetic tweaks to a rendered layout (alignment, centering, spacing, sizing, colors, or wording as it appears in the render). Trigger on requests like \"build a webpage/deck\", \"self-contained HTML\", \"center this\", \"fix the spacing/alignment\", or \"make the wording say X\" on a page you produced."
license: MIT
compatibility: designed for deepagents-code
---

# Visual Build

## Overview

Cosmetic changes to a rendered page (HTML/CSS/JS, self-contained webpages, decks) are cheap to make but impossible to confirm by reading source. Guessing — making many single edits and asking the developer to eyeball the browser between each — burns huge amounts of tokens for a one-line result. Verify against a render instead.

## Best Practices

- **Render, then reason.** Confirm every visual change against a screenshot before making the next edit or handing back. Do not infer from the source what a layout looks like.
- **Batch related edits.** Make one batch of related changes, then re-render — never one edit per turn.
- **Be honest about capability.** If you cannot render/screenshot automatically, say so up front, batch your edits, and request a single consolidated review — do not iterate one change at a time.

## Process

1. **Start a preview and capture the baseline.** Serve the page (`python3 -m http.server` from its directory, or open it in a headless browser) and screenshot the current render. Reason over the image to locate exactly what needs to change.
2. **Make a batch of related edits.** Group all edits that address the request (e.g. every rule needed to center an element) into one round rather than one `edit_file` per property.
3. **Re-render and screenshot.** Reload the served page and capture a new screenshot.
4. **Verify against the render.** Compare the screenshot to the requested change. If it is correct, stop. If not, form a specific hypothesis from what you see and return to step 2 — do not blind-guess more edits.
5. **Only then hand back.** Report the verified result. Ask the developer to eyeball the browser **only** when automated render/screenshot is unavailable, and say so explicitly.

## If headless rendering is unavailable in the sandbox

State the limitation up front. Then:

- Batch all related edits into a single round instead of iterating.
- Explain the specific change you made and the exact visual outcome you expect.
- Request one consolidated review from the developer rather than asking them to eyeball the browser after every change.

## Common Pitfalls

- **100+ blind `edit_file` guesses.** Making semantically-distinct single edits and asking the developer to check between rounds is the exact anti-pattern this skill exists to prevent — it has cost ~1.4M tokens for a one-line cosmetic result. Render and verify instead.
- **One edit per turn.** Splitting a coherent change (e.g. centering) across many turns multiplies cost with no benefit. Batch it.
- **Silent inability to render.** If you can't screenshot, do not quietly fall back to guess-and-ask — declare it and switch to batch-plus-consolidated-review.
