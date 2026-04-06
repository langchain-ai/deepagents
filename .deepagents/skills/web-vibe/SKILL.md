---
name: web-vibe
description: "Scaffold, serve, and iterate on a website from a creative prompt. Optimized for speed and visual impact. Use when: vibe coding a site, timed website challenges, or the user says 'web vibe', 'build me a website fast', 'make a landing page', 'create a webpage', 'static site from a prompt', or similar."
---

# Web Vibe

Sprint mode -- start building immediately, no planning phase. Visual impact and completeness beat code quality.

**IMPORTANT: Do NOT stop after starting the server. The server is just step 1. You MUST then rewrite the HTML, CSS, and JS to build the site described in the prompt. Keep editing files until the site is complete and polished.**

## 1. Scaffold (if no project exists yet)

If the current directory already has a `package.json` with Vite, skip to step 2.

Otherwise scaffold with Vite vanilla template:

```bash
npm create vite@latest . -- --template vanilla <<< "y"
npm install
```

**Do not use** `create-react-app`, `next`, or heavy frameworks -- too slow to scaffold.

## 2. Start dev server in background

First, detect the port. Run this command and note the output:

```bash
echo "${VIBE_PORT:-5173}"
```

Use that port number (not the variable) in all subsequent commands. Start the server fully detached so the command returns immediately:

```bash
nohup npx vite --port PORT --host > /tmp/vite.log 2>&1 & disown && sleep 2 && echo "server started"
```

Replace `PORT` with the actual number from above. The `& disown` is critical — without it the command blocks forever.

Verify it started:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:PORT
```

If you get `200`, proceed to step 3 immediately.

**Fallback** if Vite fails: `nohup npx live-server --port=PORT --no-browser > /tmp/vite.log 2>&1 & disown`

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
