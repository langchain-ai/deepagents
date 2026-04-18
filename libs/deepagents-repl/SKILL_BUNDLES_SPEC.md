# Skill modules for the QuickJS REPL

Status: draft
Owner: @hntrl
Target package: `libs/deepagents-repl` (Python)
Related: `libs/deepagents/deepagents/middleware/skills.py` (SKILL.md loader),
[deepagentsjs PR #472](https://github.com/langchain-ai/deepagentsjs/pull/472)
(precedent: swarm as a QuickJS foreign function).

## Motivation

`SkillsMiddleware` today gives the model progressive-disclosure
knowledge: a list of skills up front, full instructions pulled from
`SKILL.md` on demand. Any executable logic a skill ships — parsers,
formatters, domain helpers — has to be either (a) described in prose
and re-implemented by the model inside `eval`, or (b) shelled out to
a Python callable that the skill author has to get wired into the
agent's middleware.

quickjs-rs v0.4 ships `ModuleScope`: a per-Runtime, host-installable
ES module registry. Each skill can ship as a proper ES module the
model imports explicitly when it needs it:

```javascript
const { parse } = await import("@/skills/pdf-extract");
const pages = await parse("/data.pdf");
```

That's the full contract from the model's point of view. No `skills.*`
global, no pre-eval transform, no "invisible" lazy-load — the `import`
statement *is* the load signal, and it composes with progressive
disclosure: the model sees the skill listed, reads `SKILL.md` to
learn what's exported, then `import`s it.

The deepagentsjs swarm PR proves the broader pattern: register host
capability on the QuickJS context and document it in the system
prompt. This spec applies that to skills.

## Scope

In scope:

- Declaring a skill's module entrypoint via SKILL.md frontmatter.
- Installing each skill as a `ModuleScope` under a stable
  `@/skills/<name>` bare specifier so the model can dynamic-import it.
- Surfacing the skill's import path in the prompt listing so the model
  knows it exists and how to load it.

Out of scope:

- A `skills.*` global. The model imports; it does not reach into a
  namespace object.
- Skill-to-skill imports. One skill cannot `import … from
  "@/skills/<other>"`. Skills are leaf units of capability. If two
  skills need the same helpers, the author duplicates them into each
  skill dir or turns the shared code into a host capability. This is
  enforced structurally — see *Isolation* below.
- Cross-language parity with deepagentsjs. deepagentsjs can implement
  this independently; the spec format (frontmatter keys, specifier
  shape, exports convention) is the contract, not the code.
- Sandboxing skill code from REPL host globals. Skill modules still
  have access to `tools.*`, `readFile`, `writeFile`. Skills are
  trusted code the agent owner shipped, not adversarial.
- TypeScript type-checking. `.ts`/`.mts`/`.cts`/`.tsx` files are
  transparently stripped of type syntax by quickjs-rs (via oxidase)
  at install time; the middleware does not run `tsc`. TS parse errors
  surface at install, not at eval.
- Pre-bundling / tree-shaking. Files are installed as-is and linked
  by QuickJS.

## Design

### Directory layout

A skill directory may contain any number of `.js` / `.mjs` / `.cjs` /
`.ts` / `.mts` / `.cts` / `.jsx` / `.tsx` files alongside `SKILL.md`.
The `module` frontmatter key names the entrypoint file; other files
are helpers the entrypoint can import with relative specifiers.

If the author names the entrypoint `index.<ext>`, quickjs-rs's
subscope resolver picks it up automatically when the bare specifier
`@/skills/<name>` is imported
(`libs/quickjs-wasm/src/modules.rs:296-308` picks the first
`index.{js,mjs,cjs,ts,mts,cts,jsx,tsx}` present). If the author
names it something else, the loader renames the installed key to
`index.<ext>` at install time so the bare-specifier resolution still
works — `module: ./entry.ts` in the author's dir becomes
`"index.ts"` in the installed scope. Documented as a convention; we
recommend `index.<ext>` directly.

```
/skills/user/pdf-extract/
├── SKILL.md
├── index.ts          # entrypoint — picked automatically
├── parser.ts         # import { parsePdf } from "./parser.ts"
└── lib/
    └── utf.ts        # import { decode } from "./lib/utf.ts"
```

Frontmatter points at the entrypoint file:

```markdown
---
name: pdf-extract
description: Parse and query PDF documents
module: ./index.ts
---

# PDF extraction skill

Call with:

```javascript
const { parse, pageCount } = await import("@/skills/pdf-extract");
```

`parse(path)` returns the extracted pages…
```

The `module` frontmatter key is a path, relative to the skill
directory, to the entrypoint file. Whatever that file exports is
what the skill exports — no second declaration, no drift between
frontmatter and code. Skills that use `module` have a module
surface; skills without the key remain prose-only.

Rules on the directory contents:

- The loader enumerates the skill dir recursively via the backend
  (`backend.ls(skill_dir)`) and installs every code-extension file
  into the scope, keyed by its relative POSIX path. Non-code files
  (Markdown, JSON, images) are ignored here — skills can still reach
  them at runtime with `await readFile("/skills/.../schema.json")`.
- No enumerated file's path may escape the skill dir.
- The `module` path must resolve to one of the enumerated files. If
  the author names it `./index.ts`, the loader will have picked it
  up automatically, but we still validate the mapping at install so
  a typo surfaces cleanly.

### Extensions in import specifiers

quickjs-rs resolves relative specifiers by exact key match — no
implicit extension. `import "./parser"` does not find `"parser.ts"`.
Bare specifiers (like `@/skills/pdf-extract`) *do* auto-resolve to the
scope's `index.<ext>` in JS-first-then-TS order, so the model never
writes an extension for the import it cares about.

We considered asking the loader to install a virtual alias for every
extensionless path (register both `"parser"` and `"parser.ts"`). We
reject it: (a) it diverges from the quickjs-rs resolver's explicit
"keys match as written" contract, creating behaviour the rest of the
Python API doesn't have; (b) it only affects skill-internal code,
which the author writes once and which the model never has to
generate; (c) at most a dozen lines of author ergonomics traded
against a dozen lines of resolver magic that would have to be
maintained across quickjs-rs versions.

Concrete rules:

- The **model** writes `import … from "@/skills/<name>"`. No extension.
  quickjs-rs auto-picks the skill's `index.<ext>`.
- **Skill authors** write relative imports with extensions —
  `import { parsePdf } from "./parser.ts"`. This is documented as
  part of the skill authoring guide.
- TypeScript authors keep `.ts` extensions on both sides of the
  import (quickjs-rs strips TS type syntax but does not rewrite
  specifiers; see `libs/quickjs-wasm/src/modules.rs:340-346`).

### Isolation

Each skill is installed as its own `ModuleScope`, registered under the
bare specifier `skills/<name>` *inside* the scope we install on the
Context. The outer scope is structured so the only bare specifier the
skill's own files can resolve is nothing — it carries no subscopes
visible to the skill's code. The resolver guarantee from v0.4 is that
each scope sees only its own dict
(`libs/quickjs-wasm/spec/module-loading.md:56`), so:

- A skill's entrypoint can `import "./helper.ts"` (relative, resolves
  within its own scope).
- A skill's entrypoint cannot `import "@/skills/other"` (bare,
  would need a subscope entry in its own scope, which we don't
  provide).
- User code in the REPL can `import("@/skills/pdf-extract")` because
  the user's eval runs against the top-level scope we install on the
  Context, which *does* carry the `skills/pdf-extract` subscope.

Concretely (install shape — final key strategy TBD once the
`@/skills/…` specifier alias is pinned down):

```python
ctx.install(ModuleScope({
    "@/skills/pdf-extract": ModuleScope({
        "index.ts":       "<source>",
        "parser.ts":      "<source>",
        "lib/utf.ts":     "<source>",
        # no subscopes — the skill cannot import any other skill
    }),
    "@/skills/slugify": ModuleScope({
        "index.js":       "<source>",
    }),
    # …one entry per skill with a module frontmatter
}))
```

### The `@/` specifier convention

Model-facing code uses `@/skills/<name>` as the import specifier, not
bare `skills/<name>`. Two reasons:

1. It reads as a path root, not a bare package name, matching how
   developers write intra-project imports with path aliases. Models
   pick this up readily from training data.
2. It leaves bare `skills/<name>` available as the install-side key,
   clearly distinguishing the "install surface" from the "import
   surface."

Implementation: the middleware installs the scope under its bare key
(`skills/<name>`) and also registers the `@/skills/<name>` specifier
as a resolver-visible alias. v0.4 doesn't ship an alias API
directly; the simplest realization is to install the same scope under
both key shapes (`ModuleScope` keys are just strings; there's no cost
to having two entries that point at the same source set). Pinning
the exact implementation is an open question — see below.

### Host access

Skill modules share the Context's host globals with user code.
`tools.*`, `readFile`, `writeFile`, and anything else `REPLMiddleware`
installs are reachable inside a skill's exported function body
without any import. Same trust model as user-authored `eval` code.

### Lazy install

Installing every skill with a `module` surface eagerly at session
start is fine in the small but wastes work for skills the model
doesn't touch. We lazy-install:

- The first time the REPL encounters a static `import` of
  `@/skills/<name>` in user code, or a dynamic
  `await import("@/skills/<name>")` call, the middleware installs
  that one skill's scope (if not already installed), then lets the
  import proceed.
- Static detection is a trivial regex/AST pass over the user's eval
  input pre-submission — we're only scanning for the literal
  `@/skills/<…>` specifier. Dynamic imports with computed strings
  can't be pre-detected; they surface as an install-on-demand in the
  resolver path, which quickjs-rs v0.4 does not support. For dynamic
  calls with a non-literal specifier, the model must have already
  referenced the skill statically (or literally) earlier in the
  session so the install has happened. Documented in the prompt.
- `ctx.install()` is additive and re-install of a bare specifier is
  a silent no-op after first import
  (`libs/quickjs-wasm/tests/test_modules.py:1096-1144`), so a
  worst-case double-install races benignly.

### Thread / session lifecycle

`REPLMiddleware` allocates one `_ThreadREPL` (QuickJS Context) per
LangGraph thread, all sharing one Runtime. quickjs-rs v0.4 stores
the `ModuleScope` backing per-Runtime
(`libs/quickjs-wasm/src/runtime.rs:132-146`), so:

- **Install is per-Runtime.** First thread to reference a skill pays
  the backend fetch + TS strip + install cost. Other threads on the
  same middleware instance skip it.
- **Per-thread Context still isolates module instances.** A skill
  module's top-level `let` increments independently on each thread
  even though both Contexts draw from the same source. This matches
  how user-authored `globalThis.x` already behaves across threads
  today.
- **Reloading a changed skill mid-session doesn't propagate.**
  Re-install after first import is a silent no-op. A user editing a
  skill file during a running session won't see the change until the
  Runtime is torn down (same constraint Python-editable skills have
  today).

### System-prompt integration

`SkillsMiddleware` today renders each skill's name, description, and
SKILL.md path. When a skill has `module:`, we append one line: the
import specifier the model uses to load it.

```
- pdf-extract (path: /skills/user/pdf-extract/SKILL.md)
  Parse and query PDF documents.
  Import: `await import("@/skills/pdf-extract")`
```

That's all. Full usage — which names to destructure, what they do,
what arguments they take — is documented by the skill author inside
`SKILL.md`, which the model reads on demand through the existing
progressive-disclosure flow. This is by design:

- The skill author already writes prose usage in SKILL.md as part of
  the existing skills contract. Asking them to keep a second
  machine-extracted signature block in sync is duplicate work.
- Extracting signatures from `export` declarations is lossy —
  parameter names survive but intent and constraints don't. A doc
  comment explaining that `path` must be a POSIX path or a `Uint8Array`
  is strictly more useful than `parse(path)`.
- Less surface to maintain in the middleware: no AST walker, no
  re-export resolution, no "what does `export const x = foo()` render
  as" decision.

The loader does still validate the skill at install time — entrypoint
file exists, parses, imports resolve — so malformed skills fail
cleanly before the model tries to import them. But we don't scrape
exports to write them into the prompt.

Caching for the prompt section is trivial now: keyed on
`(skill.name, has_module)` since the only machine-generated content
is the import line, which is a pure function of the skill name.

### Error shapes

Errors surface as `EvalOutcome.error_type` values that the existing
`<error type="...">...</error>` formatter handles:

- `SkillScopeInvalid` — frontmatter `module` path doesn't resolve to
  a file in the skill dir, or the skill dir contains no code files
  at all. Raised at install attempt, before the import runs.
- `SkillInstallError` — backend fetch failed, or TS strip raised a
  parse error at `install()` time, or entrypoint resolution failed
  for structural reasons (e.g. a file path escapes the skill dir).
- `SkillLinkError` — wraps a `JSError` raised during module init:
  top-level throw in the entrypoint, or a relative import that
  doesn't match a file key. Includes the original stack.
- `SkillNotAvailable` — install previously failed on this Runtime;
  subsequent imports fail fast without re-hitting the backend.

### Backend coupling

Skill files MUST be read via the same `BackendProtocol` that
`SkillsMiddleware` uses to list skills and fetch `SKILL.md`. No new
protocol methods required. The loader uses `backend.ls(skill_dir)` to
enumerate code files and `backend.download_files([...])` to fetch
their contents in one batch; protocol errors surface as
`SkillInstallError`.

### Middleware handoff

`REPLMiddleware` and `SkillsMiddleware` don't talk to each other
today. The options:

1. **`REPLMiddleware` reads skills state.** `SkillsMiddleware` already
   writes `skills_metadata` into `AgentState` (`SkillsState` in
   `libs/deepagents/deepagents/middleware/skills.py`). Extend the
   metadata with the frontmatter `module` path (skill dir path is
   already derivable from `skill_metadata.path`). `REPLMiddleware`
   reads it off the state snapshot via `ToolRuntime`. No new
   inter-middleware API.
2. **A skills plugin interface on `REPLMiddleware`.** Explicit
   callable the user wires up. More boilerplate, more flexibility.

We recommend (1). `skills_metadata` is the existing contract and one
additional field is purely additive. `REPLMiddleware` ignores the
field if `SkillsMiddleware` isn't installed.

## Examples

### Minimal single-file skill

`/skills/user/slugify/SKILL.md`:

```markdown
---
name: slugify
description: Turn strings into URL-safe slugs
module: ./index.js
---

# Slugify

Import and call:

```javascript
const { toSlug, toSlugs } = await import("@/skills/slugify");
toSlug("Hello World");     // "hello-world"
toSlugs(["A B!", "c D"]);  // ["a-b", "c-d"]
```

`toSlug(s)` accepts any string; `toSlugs(arr)` maps over an array.
```

`/skills/user/slugify/index.js`:

```javascript
export function toSlug(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
export function toSlugs(arr) { return arr.map(toSlug); }
```

Rendered prompt entry:

```
- slugify (path: /skills/user/slugify/SKILL.md)
  Turn strings into URL-safe slugs.
  Import: `await import("@/skills/slugify")`
```

Model use (after reading SKILL.md for the export names):

```javascript
const { toSlugs } = await import("@/skills/slugify");
const out = toSlugs(["Hello World", "A B!"]);
console.log(out);  // ["hello-world", "a-b"]
```

### Multi-file skill with host access

`/skills/user/pdf-extract/`:

```
SKILL.md           (module: ./index.ts)
index.ts
parser.ts
lib/utf.ts
```

`index.ts`:

```typescript
import { parsePdf } from "./parser.ts";

export async function parse(path: string): Promise<string[]> {
  const raw = await readFile(path);
  return parsePdf(raw);
}

export function pageCount(text: string): number {
  return text.split("\f").length;
}
```

Extensions on the relative import (`"./parser.ts"`) are required.
TypeScript syntax is stripped transparently at install time. The host
`readFile` global is reachable from inside the module.

Model use:

```javascript
const { parse, pageCount } = await import("@/skills/pdf-extract");
const pages = await parse("/data.pdf");
console.log(pageCount(pages.join("\f")));
```

## Open questions

1. **Exact realization of the `@/skills/<name>` specifier alias.**
   Simplest path: install each skill's `ModuleScope` under both
   `skills/<name>` and `@/skills/<name>` keys in the top-level scope,
   since `ModuleScope` keys are just strings. This works but doubles
   the scope entry count. A cleaner alternative, if quickjs-rs adds
   a resolver alias hook, is to keep one canonical install key and
   rewrite specifiers at resolve time. Unblocking: pick the dual-key
   approach for v1; revisit if quickjs-rs grows the hook.

2. **Prose-only skills with missing `module` key.** Today's prose
   skills continue to work unchanged; the loader sees no `module`
   frontmatter and installs no scope entry. Any
   `import("@/skills/<prose-only>")` from the model fails with
   `SkillNotAvailable`. Acceptable — the prompt won't advertise an
   import for a skill without a `module` surface.

3. **Reloading edited skills.** Deferred — same constraint as
   today's Python skills. A future `ctx.reset_modules()` or Runtime
   recreation could address it.

## Rollout

1. Depend on quickjs-rs ≥ 0.4.0 in
   `libs/deepagents-repl/pyproject.toml`.
2. Implement the skill loader in `deepagents-repl`: enumerate skill
   dir, build per-skill `ModuleScope`, call `ctx.install`, track the
   install cache per-Runtime. Unit tests under
   `libs/deepagents-repl/tests/unit_tests/` cover enumeration,
   entrypoint resolution, frontmatter validation, TS extension
   handling, error paths.
3. Implement the pre-eval specifier scan: detect literal
   `import(… "@/skills/<name>" …)` and static `import … from
   "@/skills/<name>"` in user code and trigger install. No
   user-code rewriting.
4. Extend `SkillsState.skills_metadata` with the `module` path.
   `REPLMiddleware` reads it off state through `ToolRuntime`. Guard
   behind a feature flag on `REPLMiddleware.__init__` until stable.
5. Extend `SKILLS_SYSTEM_PROMPT` rendering to append the one-line
   import instruction for skills with `module:`. No signature
   extraction — authors document their API in SKILL.md prose.
6. Integration test: agent with `SkillsMiddleware` + `REPLMiddleware`,
   one multi-file skill, one prose-only skill. Verify the model can
   dynamic-import the multi-file skill, that relative imports inside
   it work, and that the prose-only skill surfaces no import
   advertisement.
7. Once stable, mirror the format in deepagentsjs (PR #472 already
   shows the session/middleware shape). The skill dir layout is
   the same contract — once the JS package exposes a `ModuleScope`
   install path, the same skill dirs load there.
