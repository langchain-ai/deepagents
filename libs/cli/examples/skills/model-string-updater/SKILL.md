---
name: model-string-updater
description: Audits and updates LLM model identifier strings (e.g. `openai:gpt-4o`, `openai:gpt-5.4`, `anthropic:claude-sonnet-4-6`, `google_genai:gemini-*`) across the deepagents monorepo — in docstrings, README/MDX files, example configs, CI matrices, hardcoded runtime defaults, and test fixtures. Use when the user asks to find, list, audit, bump, rename, deprecate, replace, or normalize model strings; add support for a new model; refresh "recommended" models in docs/examples; rotate away from a retired model; or produce a per-file report of every model reference in the repo. Trigger on phrases like "update model strings", "bump the default model", "replace gpt-4o with X", "find all openai model references", "audit model versions", "which files mention a given model", or "normalize the models used in our examples".
---

# Model String Updater

Find and update LLM model identifier strings across the deepagents monorepo consistently, without missing hidden references or accidentally changing test fixtures that assert on specific strings.

## When this skill applies

- Auditing where a given model (or provider) is referenced.
- Bumping a hardcoded default (e.g. the fallback in `config.py`).
- Refreshing docstring examples to show a current model.
- Adding/removing a model from CI matrices or eval groups.
- Deprecating a retired model across docs/examples.

## Categories of references (read before editing)

Model strings live in several semantically distinct places. Do NOT treat them uniformly — the risk/value tradeoff differs:

| # | Category | Risk of change | Typical action |
|---|----------|----------------|----------------|
| 1 | Hardcoded runtime defaults | **High** — affects real users | Change deliberately, note in PR |
| 2 | Example configs (`examples/*/deepagents.toml`, example READMEs) | Low | Keep consistent across sibling examples |
| 3 | Docstring examples in SDK/CLI source | Low | Prefer one "current recommended" string across the SDK |
| 4 | `libs/evals/MODEL_GROUPS.md` and CI matrices (`.github/workflows/*.yml`, `.github/scripts/models.py`) | Medium — changes what CI runs | Update matrix and doc together |
| 5 | Test fixtures (strings that tests assert on) | **Do not change** unless the test's intent is the model name itself | Leave alone |
| 6 | Provider-detection heuristics (e.g. prefix lists like `("gpt-", "o1", "o3", "o4", "chatgpt")`) | **High** — affects auto-detection | Only touch when adding a new family |

See `references/locations.md` for the concrete list of known locations in this repo, keyed by category.

## Workflow

Follow these steps in order. Skip a step only if it is clearly inapplicable.

### 1. Clarify the change

Before touching files, confirm with the user:

- Which model string(s) are being replaced, and with what?
- Scope: docs/examples only, runtime defaults, CI matrices, or everything?
- Does the user want the new model added alongside the old one (e.g. in an eval matrix), or a full rename?

Do not assume. A request like "bump to the new model" is ambiguous between categories 1, 2, 3, and 4 above.

### 2. Re-run the audit

The canonical locations list (`references/locations.md`) is a snapshot and may drift. Always regenerate the current list:

```bash
.venv/bin/python [YOUR_SKILLS_DIR]/model-string-updater/scripts/find_model_refs.py
```

Optional flags:

- `--pattern <regex>` — restrict to a specific model family (default matches OpenAI + Anthropic + Gemini common patterns).
- `--paths <glob> [<glob> ...]` — restrict to specific subtrees (e.g. `libs/deepagents/deepagents` for SDK only).
- `--exclude-tests` — skip `**/tests/**` and `**/test_*.py` (recommended when editing docstrings/examples).
- `--json` — emit machine-readable output for downstream processing.

Run from the repo root. The script exits non-zero only on argument errors; an empty result is a valid outcome.

### 3. Plan edits by category

Using the fresh audit output, group each hit into one of the 6 categories above. Then:

- **Category 1 (runtime defaults):** Confirm explicitly with the user before changing. Mention the change prominently in the PR body.
- **Category 2 (examples):** Keep sibling examples internally consistent (e.g. all `examples/deploy-*` should use the same provider/model family unless one is intentionally demonstrating a different one).
- **Category 3 (docstrings):** Pick ONE "current recommended" string per provider and use it uniformly. Drive-by normalization is fine while you're in the file.
- **Category 4 (CI/evals):** Update `.github/workflows/*.yml`, `.github/scripts/models.py`, and `libs/evals/MODEL_GROUPS.md` together — they must agree.
- **Category 5 (tests):** Default to no change. If a test fixture happens to use the old string but the test's intent is not the model name, you may update it, but verify the test still passes for the right reason.
- **Category 6 (heuristics):** Only change when adding a whole new model family (e.g. a new `gpt-6` or `o5` prefix). Add a test if one doesn't exist.

### 4. Apply edits

- Use targeted edits, not repo-wide find-and-replace — category 5 and 6 hits must be preserved or handled specially.
- Preserve surrounding formatting (quoting style, backticks in Markdown tables, indentation in YAML).
- When editing docstring examples, also update any adjacent prose that names the model (e.g. "defaults to `gpt-4o`").

### 5. Verify

From the repo root:

```bash
# Re-run the audit and confirm remaining hits are intentional
.venv/bin/python [YOUR_SKILLS_DIR]/model-string-updater/scripts/find_model_refs.py --exclude-tests

# Lint + format the Python packages you touched
make format
make lint

# Run tests ONLY for files you changed (CI runs the full suite)
uv run --group test pytest <specific_test_files>
```

Do not run the full test suite locally.

### 6. Write the PR

Follow the repo's `AGENTS.md` commit/PR conventions. In the PR body, call out:

- Which category of changes were made (1–6).
- Any runtime-default changes (category 1) — these deserve their own bullet.
- Any intentionally-skipped hits (e.g. test fixtures that assert on the old string).

## Known locations in this repo

See `references/locations.md` for the per-file breakdown captured from the most recent audit. Read it **before** making edits when the user's request mentions a specific file/area — it explains which hits belong to which category and which should be left alone.

## Pitfalls to avoid

- **Blanket find-and-replace on `gpt-4o`** will break ~50 test assertions in `libs/cli/tests/`. Always filter out `tests/` unless explicitly in scope.
- **Changing `config.py:1835` (`"openai:gpt-5.2"`) silently** changes the CLI's OpenAI default for every user. Surface it in the PR.
- **Editing `MODEL_GROUPS.md` without the matching workflow matrix** causes docs/CI drift.
- **Editing the prefix tuple in `config.py` (`("gpt-", "o1", "o3", "o4", "chatgpt")`)** without adding a test can break bare-name provider auto-detection.
- **Docstring model strings inside `libs/evals/tests/evals/fixtures/*.json`** are embedded copies of other docstrings — update the source, not the fixture.
