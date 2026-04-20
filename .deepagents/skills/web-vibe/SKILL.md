---
name: web-vibe
description: "Scaffold, serve, and iterate on a website from a creative prompt. Optimized for speed and visual impact. Use when: vibe coding a site, timed website challenges, or the user says 'web vibe', 'build me a website fast', 'make a landing page', 'create a webpage', 'static site from a prompt', or similar."
---

# Web Vibe

Sprint mode -- visual impact and completeness beat code quality. **You are collaborating with a human player**, not guessing in isolation -- use `ask_user` at the specific moments called out below so their taste drives the output. A site built to the player's vibe wins over a site built to yours.

**IMPORTANT: Do NOT stop after starting the server. The server is just step 1. You MUST then rewrite the HTML, CSS, and JS to build the site described in the prompt. Keep editing files until the site is complete and polished.**

## Override on `ask_user` usage

The `ask_user` tool's base description tells you to "use sparingly." **That instruction does not apply to this skill.** In web-vibe mode, collaboration is the point — use `ask_user` at the specific moments named below (direction, mid-build check-in, ambiguous follow-up). Stay within the hard caps in the time budget below.

## Time budget (5-minute round)

Budget roughly:

- ~0:00-0:20 — dev server ready + first `ask_user` fired
- ~0:20-3:30 — first complete draft
- ~3:30-4:15 — check-in + refinement
- ~4:15-5:00 — final polish, no more questions

Hard caps:

- **Maximum 3 `ask_user` calls per round**, total. Spend them on: (1) direction, (2) mid-build check-in, (3) at most ONE ambiguity clarification.
- Never issue two `ask_user` calls back-to-back without writing/editing files in between.
- If you believe more than ~60% of the round has elapsed, skip any remaining questions and keep building.

## 1. Scaffold + start dev server

Run the idempotent startup script. It creates (or reuses) a fixed round directory, copies a pre-built Vite template when available, and starts Vite detached. Safe to re-run.

```bash
bash "$VIBE_DIR/.deepagents/skills/web-vibe/start-server.sh"
```

Behavior guarantees:

- The script operates inside `$VIBE_DIR` (default `/tmp/vibe-round`). **All of your subsequent file edits MUST go inside this directory** — use absolute paths like `/tmp/vibe-round/index.html`, never `./index.html`, because your working directory may not be the round dir between tool calls.
- On success the script prints the server URL (e.g., `http://localhost:5173`) to stdout and exits 0.
- On failure it prints the tail of `/tmp/vite.log` to stderr and exits 1 — do not retry blindly; read the error, fix if possible, or tell the player the dev server is unreachable.
- Re-running is a no-op when the server is already healthy and serving the round dir. A stale server from a previous round is auto-detected and replaced.

**Do not invoke `vite`, `npm run dev`, `npm create`, or any other server yourself** — always go through the script so behavior stays consistent across rounds.

**Do not use** `create-react-app`, `next`, or heavy frameworks -- too slow to scaffold.

## 2. Ask the player for direction (ask #1 of max 3)

Immediately after the server script returns, make **one batched `ask_user` call** with 3-4 questions to lock in the creative direction. Keep questions fast (mostly multi-choice), specific to the prompt, and in plain language. Multi-choice always appends an "Other" option, so players can redirect you freely.

**Template — rewrite every question and every choice to suit the current prompt. The choices shown here are illustrative, NOT literal. For "haunted house" offer spooky palettes; for "luxury hotel" offer elegant ones; etc.**

```json
{
  "questions": [
    {
      "question": "What visual vibe should we go for? (adapt to the prompt)",
      "type": "multiple_choice",
      "choices": [
        {"value": "OPTION_A appropriate to the prompt"},
        {"value": "OPTION_B appropriate to the prompt"},
        {"value": "OPTION_C appropriate to the prompt"},
        {"value": "OPTION_D appropriate to the prompt"}
      ]
    },
    {
      "question": "Color palette direction?",
      "type": "multiple_choice",
      "choices": [
        {"value": "PALETTE_A fitting the prompt's mood"},
        {"value": "PALETTE_B fitting the prompt's mood"},
        {"value": "PALETTE_C fitting the prompt's mood"},
        {"value": "PALETTE_D fitting the prompt's mood"}
      ]
    },
    {
      "question": "Besides the hero, which section matters most?",
      "type": "multiple_choice",
      "choices": [
        {"value": "SECTION_A relevant to the prompt"},
        {"value": "SECTION_B relevant to the prompt"},
        {"value": "SECTION_C relevant to the prompt"}
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

- **Batch questions into ONE `ask_user` call.** Each call pauses the whole run.
- Prefer multi-choice over text -- one tap beats typing in a speed round.
- Ground every choice in the prompt. Never use the placeholder strings above verbatim.
- Do NOT ask what you can decide yourself (font pairings, grid layout, animation timing). Only ask about things that reflect the player's taste.
- If the player picks "Other" or types free-form, honor it literally.

## 3. Build the site (THIS IS THE MAIN WORK)

Now rewrite the project files inside `$VIBE_DIR` (absolute paths!) to match the prompt and the player's answers. This is where you spend most of your time.

**First**, clean up the Vite boilerplate:

- Rewrite `/tmp/vibe-round/index.html` completely -- replace the `<body>` content with your site's HTML
- Rewrite `/tmp/vibe-round/src/style.css` completely -- replace with your site's styles
- Rewrite `/tmp/vibe-round/src/main.js` -- delete the boilerplate, add any JS your site needs (or make it minimal)
- Delete or empty `/tmp/vibe-round/src/counter.js` -- it's Vite boilerplate

(Substitute the real `$VIBE_DIR` if the operator has overridden it.)

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

### 3a. Check in once a coherent draft exists (ask #2 of max 3)

As soon as the first complete draft is live (hero + 1-2 secondary sections, all boilerplate gone), and you still have comfortable time left (roughly ≤60% of the round elapsed), call `ask_user` once with a batched set of refinement questions. Describe what you built in one sentence and offer concrete next moves.

**Template — rewrite the description and all choices to match what you actually shipped:**

```json
{
  "questions": [
    {
      "question": "Draft is live: [ONE-SENTENCE DESCRIPTION OF WHAT YOU BUILT]. What should I push on next?",
      "type": "multiple_choice",
      "choices": [
        {"value": "CONCRETE_REFINEMENT_A based on what exists"},
        {"value": "CONCRETE_REFINEMENT_B based on what exists"},
        {"value": "CONCRETE_REFINEMENT_C based on what exists"},
        {"value": "CONCRETE_REFINEMENT_D based on what exists"}
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

After the answer, execute the chosen direction. Do not ask again unless criterion 3b fires.

### 3b. Disambiguate follow-up requests (ask #3 of max 3, only if needed)

If the player types a free-form instruction mid-build that is genuinely ambiguous ("make it cooler", "change the vibe", "fix the header"), call `ask_user` **at most once** with multi-choice options interpreting the request before editing. After that one call, commit to your best interpretation and ship.

Do NOT ask for clarification on requests that are already concrete ("make the heading red" -- just do it).

If you have already used your three `ask_user` calls or time is short, skip this and pick the most likely interpretation.

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
