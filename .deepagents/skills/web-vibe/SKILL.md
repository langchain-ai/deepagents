---
name: web-vibe
description: "Scaffold, serve, and iterate on a website from a creative prompt. Optimized for speed and visual impact. Use when: vibe coding a site, timed website challenges, or the user says 'web vibe', 'build me a website fast', 'make a landing page', 'create a webpage', 'static site from a prompt', or similar."
---

# Web Vibe

Sprint mode -- visual impact and completeness beat code quality. **You are collaborating with a human player**, not guessing in isolation -- use `ask_user` at every decision point where their taste should drive the output. A site built to the player's vibe wins over a site built to yours.

**IMPORTANT: Do NOT stop after starting the server. The server is just step 1. You MUST then rewrite the HTML, CSS, and JS to build the site described in the prompt. Keep editing files until the site is complete and polished.**

**Workflow at a glance:**

1. Start the dev server (idempotent script — fire and forget).
2. **Ask the player for direction** via one batched `ask_user` call (mostly multi-choice, ~10-15s to answer).
3. Build the first complete draft end-to-end using those answers.
4. **Check in with the player** via `ask_user` once a coherent draft exists — offer concrete refinement options.
5. Keep iterating; use `ask_user` whenever a follow-up request is ambiguous, rather than guessing.

## 1. Scaffold + start dev server

Run the idempotent startup script. It scaffolds a Vite vanilla project (if none exists in the current directory), kills any stale process on the target port, starts Vite detached, and falls back to `live-server` if Vite fails to come up. Safe to re-run.

```bash
bash /skills/web-vibe/start-server.sh
```

On success the script prints the server URL (e.g., `http://localhost:5173`) to stdout and exits 0. On failure it prints the tail of `/tmp/vite.log` to stderr and exits 1.

The script honors `$VIBE_PORT` (default `5173`) so the URL stays stable across rounds. **Do not invoke Vite, `npm create`, or `live-server` directly** — always go through the script so the behavior stays consistent.

**Do not use** `create-react-app`, `next`, or heavy frameworks -- too slow to scaffold.

## 2. Ask the player for direction

Immediately after the server script returns, make **one batched `ask_user` call** with 3-4 questions to lock in the creative direction. Keep questions fast (mostly multi-choice), specific to the prompt, and written in plain language. Multi-choice always has an auto-appended "Other" option, so players can still redirect you.

**Template — adapt choices to the prompt:**

```json
{
  "questions": [
    {
      "question": "What visual vibe should we go for?",
      "type": "multiple_choice",
      "choices": [
        {"value": "retro / 80s neon"},
        {"value": "modern minimalist"},
        {"value": "playful & hand-drawn"},
        {"value": "dark & moody"},
        {"value": "luxurious / elegant"}
      ]
    },
    {
      "question": "Color palette direction?",
      "type": "multiple_choice",
      "choices": [
        {"value": "warm (reds, oranges, yellows)"},
        {"value": "cool (blues, teals, purples)"},
        {"value": "neon / high-contrast"},
        {"value": "pastel / soft"},
        {"value": "monochrome / earth tones"}
      ]
    },
    {
      "question": "Besides the hero, which section matters most?",
      "type": "multiple_choice",
      "choices": [
        {"value": "menu / features list"},
        {"value": "gallery / visual showcase"},
        {"value": "story / about"},
        {"value": "booking / contact CTA"}
      ]
    },
    {
      "question": "Any specific detail we MUST include? (tagline, character, quirk) -- optional",
      "type": "text",
      "required": false
    }
  ]
}
```

**Rules:**

- **Batch questions into ONE `ask_user` call.** Each call pauses the run; multiple calls waste clock.
- Prefer multi-choice over text -- one tap beats typing in a speed round.
- Ground every choice in the prompt (e.g., for "haunted house" don't offer "warm & friendly" palettes -- offer tonally appropriate options).
- Do NOT ask what you can decide yourself (font pairings, grid layout, animation timing). Only ask about things that reflect the player's taste.
- If the player picks "Other" or types a free-form answer, honor it literally.

## 3. Build the site (THIS IS THE MAIN WORK)

Now rewrite the project files to match the prompt. This is where you spend most of your time.

**First**, clean up Vite boilerplate:
- Rewrite `index.html` completely -- replace the `<body>` content with your site's HTML
- Rewrite `src/style.css` completely -- replace with your site's styles
- Rewrite `src/main.js` -- delete the boilerplate, add any JS your site needs (or make it minimal)
- Delete or empty `src/counter.js` -- it's Vite boilerplate

**Then**, iterate on the design. Apply changes incrementally so Vite HMR shows progress live.

**Priority order:**

1. **Visual impact** -- colors, typography, layout, full-viewport hero section
2. **Content completeness** -- meaningful text (never lorem ipsum), real headings, real descriptions
3. **Creative touches** -- animations, gradients, illustrations, micro-interactions

**Keep going** until the site has:
- A complete layout with multiple sections
- Real, meaningful content that matches the prompt
- A cohesive color palette and typography
- At least one animation or creative touch
- No placeholder text or Vite boilerplate visible

### 3a. Check in once a coherent draft exists

As soon as the first complete draft is live (hero + 1-2 secondary sections, all boilerplate gone), **call `ask_user` again** with a single batched set of refinement questions. Describe what you just built in one sentence and offer concrete next moves. Example:

```json
{
  "questions": [
    {
      "question": "Draft is live: neon-retro hero + taco menu + footer CTA. What should I push on next?",
      "type": "multiple_choice",
      "choices": [
        {"value": "bolder hero -- bigger type, animated background"},
        {"value": "more menu polish -- imagery, hover effects, price badges"},
        {"value": "add a story / about section"},
        {"value": "add scroll-triggered animations throughout"}
      ]
    },
    {
      "question": "Anything specific to change in the copy? -- optional",
      "type": "text",
      "required": false
    }
  ]
}
```

After the answer, go execute the chosen direction. Do not ask again until another meaningful milestone lands.

### 3b. Disambiguate follow-up requests

When the player types a free-form instruction mid-build that is genuinely ambiguous ("make it cooler", "change the vibe", "fix the header"), call `ask_user` with multi-choice options interpreting the request before editing. **Cap it at one clarifying call per vague request** -- after that, commit to your best interpretation and ship.

Do NOT ask for clarification on requests that are already concrete ("make the heading red" -- just do it).

## 4. Design tips

- Pick a cohesive color palette (3-5 colors) using CSS custom properties -- don't rainbow
- Import Google Fonts via CDN `<link>` for typography variety
- Use CSS grid/flexbox, not absolute positioning
- Add subtle animations (`fade-in`, `slide-up` via CSS `@keyframes`) -- high impact, low effort
- Full-viewport hero sections photograph well for judging
- Use `min-height: 100vh; min-height: 100dvh;` for hero sections (dvh handles mobile browser chrome)
- Add `box-sizing: border-box` globally

## 5. Judging criteria

The site is screenshotted and judged by an LLM on:

- **Visual design** -- layout, color, typography, polish
- **Content completeness** -- meaningful text, no placeholders, matches the prompt
- **Creativity / wow factor** -- animations, unique touches, personality
- **Prompt adherence** -- does the site match what was asked for?

Optimize for screenshot appeal -- a single frame is all the judge sees.
