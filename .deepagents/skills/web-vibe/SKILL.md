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

- ~0:00-0:10 — server already up (handled pre-turn); fire first `ask_user`
- ~0:20-3:30 — first complete draft
- ~3:30-4:15 — check-in + refinement
- ~4:15-5:00 — final polish, no more questions

Hard caps:

- **Maximum 3 `ask_user` calls per round**, total. Spend them on: (1) direction, (2) mid-build check-in, (3) at most ONE ambiguity clarification.
- Never issue two `ask_user` calls back-to-back without writing/editing files in between.
- If you believe more than ~60% of the round has elapsed, skip any remaining questions and keep building.

## 1. Dev server (already running)

The round launcher starts the Vite dev server **before** your first turn via the CLI's `--startup-cmd` flag, which runs the idempotent `start-server.sh` script. By the time you read this, the server URL (e.g., `http://localhost:5173`) is already in the transcript above — skim for it and move on.

**Only re-run the script if** the startup output shows a failure (the script prints `ERROR:` + a log tail to stderr and exits non-zero). To retry:

```bash
bash "$VIBE_DIR/.deepagents/skills/web-vibe/start-server.sh"
```

The script is idempotent and safe to re-run. Otherwise, skip straight to step 2 — do **not** re-run it "just to check".

Ground rules for the rest of the round:

- The script operates inside `$VIBE_DIR`. **All of your subsequent file edits MUST go inside this directory** — always use `$VIBE_DIR/...` absolute paths (e.g., `$VIBE_DIR/index.html`), never `./index.html`, because your working directory may not be the round dir between tool calls.
- **Never call `write_file` on a path that already exists — it will refuse.** The Vite template scaffolds these files up front: `$VIBE_DIR/index.html`, `$VIBE_DIR/src/main.js`, `$VIBE_DIR/src/style.css`, `$VIBE_DIR/src/counter.js`, `$VIBE_DIR/package.json`. To change any of them, **read first, then `edit_file`**. Use `write_file` only for genuinely new paths (e.g., a new component file you are adding). If you hit the "already exists" error, stop guessing — switch to `edit_file` on that exact path.
- **Do not invoke `vite`, `npm run dev`, `npm create`, or any other server yourself** — always go through the script so behavior stays consistent across rounds.
- **Do not use** `create-react-app`, `next`, or heavy frameworks -- too slow to scaffold.

## 2. Ask the player for direction (ask #1 of max 3)

As your very first action of the round, make **one batched `ask_user` call** with 3-4 open-ended questions to lock in the creative direction. The server is already up — don't waste a turn confirming it. Keep questions short, specific to the prompt, and in plain language. **Do not pre-fill answer options — let the player write what they actually want.**

**Template — ask the player in their own words. Do NOT suggest example answers inside the question text. No `e.g.`, no "(dark/playful/minimal)", no "etc." trailing a list. If the question is unclear on its own, reword the question itself — do not patch it with a parenthetical of examples.**

```json
{
  "questions": [
    {
      "question": "What visual vibe are we going for?",
      "type": "text",
      "required": true
    },
    {
      "question": "Color palette direction?",
      "type": "text",
      "required": true
    },
    {
      "question": "Besides the hero, which section matters most?",
      "type": "text",
      "required": true
    },
    {
      "question": "Any specific detail we MUST include? (tagline, character, quirk) -- optional",
      "type": "text",
      "required": false
    }
  ]
}
```

**Rules (apply to EVERY `ask_user` call in this skill, not just this one):**

- **Batch questions into ONE `ask_user` call.** Each call pauses the whole run.
- **Never suggest answers.** No multi-choice, no "e.g. ...", no "(dark/playful/minimal)", no trailing "etc." The player's taste drives the output — don't anchor them to yours. The instant you type `e.g.` or `(` after the `?`, stop and rewrite.
- Do NOT ask what you can decide yourself (font pairings, grid layout, animation timing). Only ask about things that reflect the player's taste.
- Honor free-form answers literally, even if unusual.

**Examples:**

- ✅ `"What should I push on next?"`
- ❌ `"What should I push on next? (e.g. make it grungier, change copy, different vibe)"` — lists choices
- ✅ `"Color palette direction?"`
- ❌ `"Color palette direction? (dark, playful, minimal)"` — lists choices
- ✅ `"Which section matters most after the hero?"`
- ❌ `"Which section matters most? e.g. features, pricing, about"` — lists choices

## 3. Build the site (THIS IS THE MAIN WORK)

Now rewrite the project files inside `$VIBE_DIR` (absolute paths!) to match the prompt and the player's answers. This is where you spend most of your time.

**First**, clean up the Vite boilerplate. These files **already exist** from the template — you MUST `read_file` then `edit_file`. Do NOT `write_file` on any of them; it will refuse and burn a turn.

- `edit_file` `$VIBE_DIR/index.html` -- swap the `<body>` content for your site's HTML
- `edit_file` `$VIBE_DIR/src/style.css` -- swap in your site's styles
- `edit_file` `$VIBE_DIR/src/main.js` -- drop the boilerplate, add only the JS your site needs
- `edit_file` `$VIBE_DIR/src/counter.js` -- empty it out; it's Vite boilerplate

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

As soon as the first complete draft is live (hero + 1-2 secondary sections, all boilerplate gone), and you still have comfortable time left (roughly ≤60% of the round elapsed), call `ask_user` once with a batched set of refinement questions. Describe what you built in one sentence, then ask the player what to push on.

**The no-suggestions rule from section 2 applies here too.** No `e.g.`, no parenthetical list of directions, no "etc." after the `?`. Describe the draft, then ask an open question and stop.

**Template — rewrite the description to match what you actually shipped:**

```json
{
  "questions": [
    {
      "question": "Draft is live: [ONE-SENTENCE DESCRIPTION OF WHAT YOU BUILT]. What should I push on next?",
      "type": "text",
      "required": true
    },
    {
      "question": "Anything specific to change in the copy? -- optional",
      "type": "text",
      "required": false
    }
  ]
}
```

After the answer, execute the player's direction. Do not ask again unless criterion 3b fires.

### 3b. Disambiguate follow-up requests (ask #3 of max 3, only if needed)

If the player types a free-form instruction mid-build that is genuinely ambiguous ("make it cooler", "change the vibe", "fix the header"), call `ask_user` **at most once** with a short text question asking them to clarify what they mean before editing. Do not suggest interpretations — let them say it in their own words. After that one call, commit to your best interpretation and ship.

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
