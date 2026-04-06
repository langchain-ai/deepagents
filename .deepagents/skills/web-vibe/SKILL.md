---
name: web-vibe
description: "Scaffold, serve, and iterate on a website from a creative prompt. Optimized for speed and visual impact. Use when: vibe coding a site, timed website challenges, or the user says 'web vibe', 'build me a website fast', 'make a landing page', 'create a webpage', 'static site from a prompt', or similar."
---

# Web Vibe

Sprint mode -- start building immediately, no planning phase. Visual impact and completeness beat code quality.

## 1. Scaffold (first 30 seconds)

Create a fresh directory for the project, then scaffold with Vite vanilla template:

```bash
mkdir project && cd project
npm create vite@latest . -- --template vanilla
npm install
```

**Do not use** `create-react-app`, `next`, or heavy frameworks -- too slow to scaffold.

**Fallback** if Vite fails:

Create a single `index.html` and serve with:

```bash
npx live-server --port=${VIBE_PORT:-5173} --no-browser
```

## 2. Start dev server immediately

Read `VIBE_PORT` from the environment. Default to `5173` if unset. Bind to all interfaces so LAN devices can reach it:

```bash
npx vite --port ${VIBE_PORT:-5173} --host
```

Start the server **within the first minute**. Use the Bash tool with `run_in_background: true` (or append `&`) and continue editing.

## 3. Iteration loop

Apply changes incrementally so Vite HMR shows progress live. Focus edits on `index.html` and `style.css`. Delete or empty `counter.js` and simplify `main.js` -- the Vite boilerplate JS is irrelevant.

**Priority order:**

1. **Visual impact** -- colors, typography, layout, full-viewport hero section
2. **Content completeness** -- meaningful text (never lorem ipsum), real headings, real descriptions
3. **Creative touches** -- animations, gradients, illustrations, micro-interactions

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
