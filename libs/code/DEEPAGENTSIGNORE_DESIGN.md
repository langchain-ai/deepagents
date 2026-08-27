# `.deepagentsignore` design for Deep Agents Code

- **Status:** Proposed
- **Scope:** `libs/code`
- **Related:** [deepagents#2143](https://github.com/langchain-ai/deepagents/issues/2143), [OpenWiki#165](https://github.com/langchain-ai/openwiki/pull/165)

## Executive summary

Implement `.deepagentsignore` in Deep Agents Code (`dcode`), not as a new Deep Agents SDK feature.

The file should reduce irrelevant or sensitive project files entering the model context through dcode-owned surfaces such as file tools, `@` mentions, autocomplete, attachments, local context, and rubric files. One dcode-owned matcher should supply the same answer everywhere.

This feature must **not** be presented as a security boundary while `execute` is available. Shell commands, interpreters, build tools, hooks, MCP servers, and user-approved extensions can read any file available to their environment. OpenWiki avoids that bypass by almost entirely disabling shell execution whenever `.openwikiignore` is active; that tradeoff is incompatible with a general-purpose coding agent.

### Recommendation

- Ship `.deepagentsignore` as a **context and agent-file-tool exclusion**, not a secret vault.
- Keep the implementation in `libs/code`.
- Reuse the SDK's backend protocol, but do not add `.deepagentsignore` naming or policy to the SDK.
- Apply rules to all dcode-controlled reads, discovery, attachments, and direct file mutations.
- Leave `execute` behavior unchanged and document the limitation prominently.
- Use a mature gitignore parser rather than maintaining a partial parser in dcode.

## Why this belongs in dcode

The SDK already provides generic filesystem permissions in `FilesystemMiddleware`. Those permissions intentionally reject execution-capable backends because a filesystem rule cannot constrain arbitrary shell commands.

`.deepagentsignore` adds product policy beyond a generic permission primitive:

- a dcode-specific filename;
- profile and project rule precedence;
- default coding-project exclusions;
- behavior for dcode's chat input and TUI;
- behavior for dcode's local-context prompt;
- a product decision about shell access and user messaging.

Those decisions belong to the coding-agent product. Other SDK consumers may want different filenames, defaults, trust models, or shell behavior.

The SDK may later gain more general filtering hooks, but that is not required to define or ship this feature.

## User experience

A project can add a file at its root:

```gitignore
# Keep credentials and generated output out of agent file operations
.env
secrets/
dist/

# This example is safe for the agent to inspect
!secrets/example.env
```

A user can also define profile-wide rules at:

```text
$DEEPAGENTS_HOME/.deepagentsignore
```

`DEEPAGENTS_HOME` defaults to `~/.deepagents`. Dcode must use its resolved profile path rather than hard-coding the default.

### Rule precedence

Rules are evaluated in this order:

1. dcode defaults;
2. profile rules;
3. project rules.

The last matching rule wins. A later `!` rule can re-include a path excluded by an earlier rule.

Suggested defaults:

```gitignore
.git/
node_modules/
.venv/
venv/
__pycache__/
dist/
build/
```

Defaults should be short and unsurprising. Language-specific caches can be added later based on observed noise rather than starting with a large deny list.

### Expected behavior

| Surface | Ignored path behavior |
| --- | --- |
| `read_file` | Return a clear exclusion error without reading the file. |
| `ls` | Reject an ignored target directory; otherwise omit ignored entries. |
| `glob` | Reject an ignored search root; otherwise omit ignored matches. |
| `grep` | Reject an ignored search root; otherwise omit matches from ignored files. |
| `write_file`, `edit_file`, `delete` | Return a clear exclusion error. |
| Backend upload/download | Return `permission_denied` for excluded items while preserving batch result order. |
| `@file` parsing | Do not attach the file; show a concise warning. |
| `@` autocomplete | Do not suggest the path. |
| Drag-and-drop or pasted file attachment | Reject the ignored attachment with a concise warning. |
| `/rubric file` | Refuse to load an ignored project file. |
| Local-context file list/tree/Makefile | Do not place ignored paths or contents in the system prompt. |
| `execute` | Unchanged; rules do not constrain shell commands. |
| Hooks, MCP tools, skills, custom tools | Unchanged; these are outside the file-tool filter. |

Blocking direct mutations makes the behavior easy to explain: dcode's file tools do not touch ignored paths. It still does not make the path immutable because `execute` and extensions remain outside the filter.

### User-facing wording

Use wording such as:

> `.deepagentsignore` keeps matching project paths out of dcode's built-in file tools and context. It does not prevent shell commands or extensions from accessing those paths. Use OS permissions or an appropriately configured sandbox for isolation.

Avoid wording such as “secure,” “protected,” “private,” or “cannot be read.”

## OpenWiki precedent

OpenWiki implements `.openwikiignore` with two main pieces:

1. A shared ordered matcher supports comments, `!` negation, root anchoring, directory rules, `*`, `**`, and `?`. Paths are normalized before matching, and matching is case-insensitive to prevent alternate-casing bypasses on case-insensitive filesystems.
2. An `OpenWikiLocalShellBackend` subclass checks every backend operation. It denies direct reads and mutations, filters `ls`/`glob`/`grep`, and handles file transfers.

OpenWiki also filters git-derived context, source fingerprints, and evidence resolution. Most importantly, it restricts shell execution to a tiny allowlist—currently maintenance commands such as `pwd` and `git rev-parse HEAD`—when ignore rules are active.

The reusable idea is **one matcher enforced at every context ingress**. The OpenWiki code should not be copied directly:

- it is TypeScript against a different runtime;
- its custom parser implements only a subset of gitignore syntax;
- its shell lockdown is reasonable for a documentation agent but not for dcode.

## Proposed architecture

```text
                         profile rules
                              +
 dcode defaults ------> IgnoreSpec <------ project rules
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   filtered backend     input/attachments     local context
   file operations      and autocomplete      prompt filtering
          |
          v
   CompositeBackend
   + artifact routes
          |
          v
 main agent, local subagents, rubric/criteria readers

 execute ------------------------------------------------> unchanged backend
```

### 1. `IgnoreSpec`: one source of truth

Add a small dcode module responsible for:

- loading defaults, profile rules, and project rules;
- parsing gitignore-compatible syntax;
- normalizing candidate paths relative to the active project root;
- answering `is_ignored(path, is_directory=False)`;
- filtering path collections without changing their order;
- retaining source information for diagnostics.

Load the rules once per project/session and pass the resulting immutable object to consumers. Do not re-read ignore files for every tool call or autocomplete query.

When dcode changes project directories, rebuild the project portion and invalidate the autocomplete and local-context caches. Profile rules stay tied to the process's resolved `DEEPAGENTS_HOME`.

### 2. Parser semantics

The issue requests standard gitignore behavior. The recommended implementation is a direct `pathspec` dependency using its GitWildMatch implementation, subject to normal dependency review.

Reasons:

- negation ordering and directory semantics have non-obvious edge cases;
- escaped `#` and `!`, character classes, anchoring, and globstars are easy to implement incorrectly;
- OpenWiki's hand-written parser is intentionally partial;
- sharing a well-tested parser reduces divergence from user expectations.

If adding `pathspec` is rejected, narrow and document the supported syntax rather than claiming full gitignore compatibility.

Dcode should match case-insensitively. This can over-exclude two differently cased paths on a case-sensitive filesystem, but it prevents an exclusion from behaving differently when the same checkout moves between Linux, default macOS, and Windows. Over-exclusion is the safer failure mode for a context filter.

### 3. Path handling

Every check uses a project-relative normalized path. The implementation should:

1. validate agent-supplied virtual paths with the SDK's `validate_path` before access;
2. normalize separators to `/`;
3. collapse equivalent `.` segments without permitting `..` traversal;
4. retain the lexical path for matching even when it is a symlink;
5. let the underlying backend perform its existing containment and symlink checks;
6. treat paths outside the project root as outside `.deepagentsignore` scope.

A matching failure should fail closed for the individual dcode operation: return an exclusion error rather than silently reading the file. A malformed or unreadable ignore file should stop agent startup with its source path in the error. Silently dropping rules would make behavior look enabled when it is not.

### 4. Backend filter

Wrap the active **project backend before it becomes the default route of `CompositeBackend`**. This keeps dcode's artifact and conversation-history routes outside project rules while centralizing project file operations.

The filter must preserve the complete backend contract:

- sync and async variants of `ls`, `read`, `glob`, `grep`, `write`, `edit`, and `delete`;
- sync and async upload/download methods;
- `grep`'s `max_count` behavior and `truncated` state;
- `GlobResult` truncation metadata;
- partial-success ordering for batch transfers;
- sandbox `id`, `execute`, `aexecute`, and timeout behavior when the wrapped backend supports execution.

The wrapper delegates `execute` unchanged. It must not inspect command strings and pretend to enforce path rules; shell syntax, subprocesses, variables, interpreters, and filesystem aliases make such a deny list bypassable.

#### Capability caveat

`FilesystemMiddleware` currently detects capture-at-source support with a concrete `BaseSandbox` check. A generic proxy around a remote `BaseSandbox` will no longer satisfy that check, so large shell output falls back to generic post-execution eviction.

Options, in preference order:

1. preserve the capability through a small generic SDK protocol change, without adding `.deepagentsignore` policy to the SDK;
2. explicitly accept generic eviction for filtered remote backends in v1 and cover it with regression tests;
3. defer remote-backend support until the capability can be preserved.

Do not silently lose large-output bounds or claim exact parity without testing this path.

### 5. Agent and subagent coverage

The main agent and local subagents share the composite backend, so filtering its project-default route gives both the same behavior. Existing `--allow-fs-tools` middleware injection remains unchanged.

Two additional repository readers need explicit attention:

- the goal-criteria agent;
- the rubric grader's separate repository backend.

Any separately constructed repository backend must receive the same `IgnoreSpec` and filter. Artifact-only readers must not receive project rules.

Remote async subagents run in external deployments with their own tools and trust boundary. Dcode cannot enforce local `.deepagentsignore` rules inside them; document them as out of scope unless their API later accepts an equivalent policy.

### 6. Input and attachment coverage

Several dcode paths read files directly with `Path`, bypassing the agent backend. Check the shared `IgnoreSpec` before these reads:

- `parse_file_mentions()` and `_read_mentioned_file()`;
- pasted and drag-and-drop paths from `parse_pasted_path_payload()`;
- image and video attachment loading;
- `/rubric file`.

Autocomplete should filter `_get_project_files()` after tracked and untracked results are combined. Git's own excludes are not sufficient because `.deepagentsignore` is independent from `.gitignore`.

An explicitly selected ignored file should be rejected rather than silently omitted. The warning should identify the path and `.deepagentsignore`, but should not reveal file content or rule details unless diagnostics are requested.

### 7. Local context

`LocalContextMiddleware` currently executes a static detection script that can read a directory tree and the first lines of a `Makefile`. This path bypasses backend file-tool filtering.

The dynamic ignore patterns must **not** be interpolated into a shell command. That would create a command-injection surface and would still implement gitignore semantics inconsistently.

Preferred approach:

- keep runtime and git metadata collection in the bounded static script;
- collect or post-process model-visible file/tree/Makefile sections using structured Python data and the shared matcher;
- skip a content section entirely when its source path is ignored;
- preserve the existing time, line, and output caps.

Local context is cached. Recompute it when the project or effective ignore rules change.

## Security model

### What this feature does

- reduces accidental context ingestion through dcode-controlled paths;
- reduces noisy discovery results and unnecessary model tokens;
- gives teams a shared convention for files the coding agent should not touch directly;
- applies consistently to local file tools and dcode's own UI readers.

### What this feature does not do

It does not stop access through:

- `execute`, including `cat`, Python, Git object reads, shell expansion, or build tools;
- project code run by the user or agent;
- hooks and MCP servers;
- custom tools or skills;
- remote async subagents;
- other local processes running as the same user.

For actual isolation, use filesystem permissions, remove shell capability, or run dcode in a sandbox whose filesystem policy is enforced below the shell.

### Why not copy OpenWiki's strict shell policy?

OpenWiki generates documentation and can work with four constrained file tools. Dcode is a coding agent: tests, builds, package managers, Git, formatters, and debuggers are core workflows. Restricting shell to `pwd` and `git rev-parse HEAD` would make dcode largely unusable whenever defaults or a project ignore file are active.

A future strict mode could remove `execute` and other extension surfaces for specialized environments, but it should be a separate feature with an explicit threat model—not an implied guarantee of `.deepagentsignore`.

## Failure behavior

| Failure | Behavior |
| --- | --- |
| Ignore file is absent | Continue with defaults and any rules from the other scope. |
| Ignore file is malformed | Stop startup and identify the file and parse error. |
| Ignore file cannot be read | Stop startup; do not silently disable its rules. |
| Candidate path cannot be normalized | Reject that operation without touching the backend. |
| Result filtering fails | Return an operation error; do not return unfiltered results. |
| Rules change during a session | Take effect after project switch or explicit reload; do not partly update caches. |

This mirrors dcode's existing preference for one coherent configuration generation over partially live configuration.

## Testing strategy

### Matcher tests

Cover:

- comments and blank lines;
- escaped `#` and `!`;
- root anchoring;
- directory-only patterns;
- `*`, `**`, `?`, and character classes;
- last-match-wins negation across defaults, profile, and project rules;
- separator and case normalization;
- `.`/`..` and symlink spellings;
- missing, malformed, and unreadable files.

### Backend contract tests

Run the same behavioral cases against:

- local `FilesystemBackend` without shell;
- local `LocalShellBackend`;
- an async test backend;
- a representative remote `BaseSandbox` test double;
- `CompositeBackend` with project-default and artifact routes.

Verify both sync and async calls, all result metadata, batch ordering, timeout forwarding, and that `execute` remains available but bypasses project rules by design.

### Product-surface tests

Verify ignored paths never appear through:

- `@` autocomplete;
- `@file` embedding;
- pasted or dropped text, image, and video files;
- `/rubric file`;
- local-context file list, tree, or Makefile content;
- the main agent, local subagents, goal-criteria agent, and rubric grader.

Also verify project switching and reload invalidate every relevant cache.

### Security regression tests

Add explicit tests proving the documented limitation: a shell command can still read an ignored path. This prevents a future UI or documentation change from accidentally upgrading the claim to a security guarantee.

## Rollout

1. Land the matcher, loader, and diagnostics with focused tests.
2. Add backend filtering for local no-shell and local-shell modes.
3. Cover direct TUI reads, attachments, autocomplete, and local context.
4. Add rubric/criteria coverage and remote sandbox parity.
5. Update user documentation and the threat model.
6. Consider enabling defaults only after measuring unexpected exclusions in development builds.

A single release can contain all steps, but keeping the implementation in these reviewable slices makes bypasses easier to spot.

## Alternatives considered

### Put `.deepagentsignore` in the SDK

Rejected. The SDK should expose generic capabilities, not a dcode filename, defaults, global path, TUI semantics, or shell UX. It also cannot turn path filtering into a shell security boundary.

### Use `FilesystemMiddleware._permissions`

Rejected for dcode's main backend. The API is private and correctly raises `NotImplementedError` for execution-capable backends when rules affect the executable default route. Using it only in no-shell mode would produce inconsistent behavior.

### Copy OpenWiki's matcher and backend verbatim

Rejected. The parser is partial, the language and backend contracts differ, and OpenWiki's shell policy does not fit dcode.

### Parse command strings and block references to ignored paths

Rejected. Shell aliases, variables, substitutions, interpreters, Git object access, symlinks, and subprocesses make string filtering bypassable and likely to create false confidence.

### Treat `.gitignore` as `.deepagentsignore`

Rejected. Git tracking policy and model-context policy are different. A tracked fixture may still be inappropriate for model context, while an untracked source file may be exactly what the user wants the agent to inspect.

## Open decisions

1. **Dependency:** approve `pathspec`, or intentionally document a narrower syntax.
2. **Defaults:** enable the suggested defaults in the first release or start with user-authored rules only.
3. **Remote sandboxes:** accept generic large-output eviction in v1, make the small SDK capability change, or defer remote support.
4. **Reload:** integrate with `/reload` immediately or apply changes on the next session/project switch first.
5. **Strict mode:** leave for a separate proposal unless a concrete shell-free use case requires it now.

## Acceptance criteria

The design is implemented when:

- one immutable ruleset drives every dcode-controlled project-file context surface;
- local main-agent and subagent file tools deny or filter ignored paths consistently;
- the rubric/criteria readers and remote mode have an explicit, tested behavior;
- artifact and conversation-history routes still work;
- ignore changes cannot yield mixed old/new caches;
- documentation clearly states that `execute` and extensions can bypass the filter;
- no SDK API or documentation implies that filesystem path rules constrain arbitrary shell execution.
