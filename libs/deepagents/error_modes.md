# Common eval error modes observed

- **Planner detours (verification/discovery):** When the prompt leaves room for interpretation, the model often inserts extra "confidence-building" tool calls (e.g., `ls` to orient, `read_file` to verify a write, `read_file` after an `edit_file`) even if the task is already determinate from the harness context.

- **Action vs. permission-seeking:** If the request implies a potential policy conflict (renaming a user-specified path, editing without inspection, using a different tool than requested), some models switch from executing to asking for approval, which collapses single-turn evals that expect immediate tool use.

- **Edit safety heuristics are inconsistent across tasks/models:** Even with a precise replacement, the model may (a) read first because it assumes "must inspect before modifying," (b) explain that constraint in an extra final step, or (c) attempt the edit directly. This depends on perceived risk and on whether the harness/tooling implicitly encourages read-before-edit.

- **"Efficiency" is tool-dependent unless forbidden:** In "find the right file" tasks, models may treat `grep` as the most efficient default even when the eval expects parallel `read_file` over enumerated files; unprompted efficiency therefore becomes "use search" unless the prompt explicitly forbids it.

- **Output formatting can be semantically correct but string-unequal:** Models sometimes emit paths with formatting/backticks or zero-width Unicode characters, causing strict substring/equals assertions to fail despite visually correct answers.
