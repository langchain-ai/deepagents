# Eval failures analysis (openai:gpt-5.1-codex)

This document explains, test by test, why each eval failed (based on the provided traceback) and lists potential fixes.

## tests/evals/test_file_operations.py::test_tool_error_recovery_read_file_then_ls

**Why it failed**

- The eval expects a specific recovery pattern:
  1) `read_file` on the misspelled path (`/docs/releasse_notes.md`)
  2) `ls /docs` after the read fails
  3) `read_file` the correct file (`/docs/release_notes.md`)
  4) final answer
- The model instead chose the more direct strategy:
  - step 1: `ls /docs`
  - step 2: `read_file /docs/release_notes.md`
  - step 3: answer
- This produces `num_agent_steps=3` vs expected `4`.

**Potential fixes**

- Tighten the prompt to force the intended pattern: "First attempt to read `/docs/releasse_notes.md` exactly; if it fails, then list `/docs` to find the correct file; then read it."
- Or loosen the eval to accept either valid strategy ("try read then recover" OR "ls first").

---

## tests/evals/test_file_operations.py::test_write_files_in_parallel

**Why it failed**

- The eval expects 2 steps:
  1) parallel `write_file` to `/a.md` and `/b.md`
  2) final reply
- The model added a verification step:
  - step 1: writes
  - step 2: reads `/a.md` and `/b.md` (parallel)
  - step 3: final text confirming contents

**Potential fixes**

- Tighten the prompt: "Do NOT read any files after writing. Reply DONE only." and keep strict 2-step expectation.
- Or loosen the eval to allow optional verification reads (accept 2 or 3 steps; 2 or 4 tool calls).

---

## tests/evals/test_file_operations.py::test_edit_file_replace_text

**Why it failed**

- The eval expects the agent to comply with: "Do not read the file before editing it."
- The model refused and asked permission to read first, producing only a single text step (no tool calls).
- That yields `num_agent_steps=1` vs expected `2`.

**Potential fixes**

- If the goal is strict compliance, tighten the prompt to remove perceived safety concerns, e.g. "Use `edit_file` with `replace_all=true` to replace the exact string `cat` with `dog`. Do not ask questions."
- If the goal is to allow cautious behavior, revise the eval to accept a clarification question as a valid first step.
  - Note: in a single-turn harness, accepting questions can make the test non-deterministic unless you treat the question as a pass condition.

---

## tests/evals/test_file_operations.py::test_find_magic_phrase_deep_nesting

**Why it failed**

- The eval expects 2 steps: `grep` then answer.
- The model did: `grep` -> `read_file` of the matching file -> answer.
- This yields 3 steps instead of 2.

**Potential fixes**

- Tighten prompt to forbid read verification: "Use `grep` to locate the phrase value; do not read any files; reply with the value only."
- Or loosen eval to allow an optional `read_file` verification step.

---

## tests/evals/test_file_operations.py::test_identify_quote_author_from_directory_unprompted_efficiency

**Why it failed**

- The eval expects:
  1) `ls /quotes`
  2) read all quote files (ideally in parallel)
  3) answer with the path
- The model instead:
  - used `grep` for "Grace Hopper" (not required; and may be disallowed depending on prompt),
  - read only a subset of files,
  - answered after 5 steps.

**Potential fixes**

- Tighten prompt: explicitly forbid grep and require reading all files: "List `/quotes`, then read all quote files in parallel. Do not use grep. Reply with the file path only."
- Strengthen expectations to require all 5 `read_file` calls in the same step.
- If you truly want "unprompted efficiency", you likely still need to forbid grep explicitly; otherwise many models will reasonably choose grep as the most efficient tool.

---

## tests/evals/test_memory.py::test_memory_guided_behavior_naming_convention

**Why it failed**

- The model responded with a clarification question / proposal about a naming convention ("Could I create `/config_api.txt` instead?") rather than performing the expected action sequence.
- This collapses the run to a single text step.

**Potential fixes**

- Tighten prompt to remove negotiation: "Create `/api.txt` exactly; do not rename; do not ask questions."
- Or update the eval to accept a clarification question as an allowed outcome (again, single-turn makes this tricky).

---

## tests/evals/test_memory.py::test_memory_influences_file_content

**Why it failed**

- The eval expected 2 steps, but the model inserted an extra exploratory `ls /` before writing `/add.py`.
- That yields 3 steps.

**Potential fixes**

- Tighten prompt: "Do not list directories; just write `/add.py`..."
- Or loosen expectation to allow an optional preliminary `ls`.

---

## tests/evals/test_skills.py::test_combine_two_skills

**Why it failed**

- The eval expects 2 steps: read both skills (parallel) then answer.
- The model performed multiple `ls` calls to discover skill paths and then read the skills, producing 6 steps.

**Potential fixes**

- Tighten prompt to specify the exact files and forbid `ls`:
  - "Read `/skills/user/frontend-deploy/SKILL.md` and `/skills/user/backend-deploy/SKILL.md` in parallel; do not list directories."
- Or loosen eval to allow discovery via `ls` (but step counts will vary widely across models).

---

## tests/evals/test_skills.py::test_update_skill_typo_fix_no_read

**Why it failed**

- The eval expects the agent to edit without reading first.
- The model refused, claiming it must read before editing, and asked for confirmation.
- Result: 1 step (text) instead of expected 2.

**Potential fixes**

- Align the test with the harness/tool constraints: require `read_file` then `edit_file` (and optionally a final reply).
- If you want to keep "no read" as the instruction, treat refusal as an acceptable outcome, or rewrite the goal of the test to be about policy compliance rather than file mutation.

---

## tests/evals/test_skills.py::test_update_skill_typo_fix_requires_read

**Why it failed**

- The eval expects 3 steps: read -> edit -> confirm.
- The model performed multiple edits (first edit attempt used an `old_string` that appears to include a line-number prefix, then tried again), then re-read to verify, then responded.
- That yields 5 steps.

**Potential fixes**

- Tighten prompt to reduce mismatch risk:
  - specify the exact substring to replace (without line numbers)
  - suggest `replace_all=true` when appropriate
- Loosen expectations to allow multiple `edit_file` attempts and an optional verification read.

---

## tests/evals/test_skills.py::test_find_skill_in_correct_path

**Why it failed**

- The eval expects 3 steps: read -> edit -> confirm.
- The model added a post-edit `read_file` verification, producing 4 steps.

**Potential fixes**

- Tighten prompt: "Do not re-read after editing; reply DONE only."
- Or loosen eval to allow an optional verification read.

---

# Cross-cutting patterns

1) **Step-count brittleness**: Many models add optional verification (`read_file`) or discovery (`ls`) steps. If those are acceptable, allow them.

2) **Single-turn + clarification questions**: Some models respond with questions instead of acting. If you want robustness, add "Do not ask questions" and make file paths non-negotiable.

3) **Edit-without-read conflicts**: If the harness/tooling requires a prior read, an eval that demands "edit without reading" will be model-dependent and often fail.
