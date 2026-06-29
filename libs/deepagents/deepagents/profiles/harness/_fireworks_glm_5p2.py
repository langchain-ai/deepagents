"""Built-in GLM-5.2 harness profile for Fireworks.

GLM-5.2 is used in the eval harness as a text-oriented model. The filesystem
tooling may encounter media files even for terminal-oriented tasks, so this
profile nudges the model toward inspecting media through the sandbox instead
of relying on direct visual input from the chat model.
"""

from deepagents.profiles.harness._fireworks_glm_5p2_middleware import (
    FinalizeMiddleware,
    RambleMiddleware,
)
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_SYSTEM_PROMPT_SUFFIX = """\
<media_file_handling>
This model profile does not support direct image input through `read_file`.
Do not call `read_file` on image or video files. When image or video files are
relevant to the task, inspect them with shell commands or scripts in the
sandbox using the file path, for example Python image-processing, OCR,
metadata, or frame-extraction utilities, rather than asking the chat model to
view the media directly.
</media_file_handling>

<verification_discipline>
Before treating a task as done:

- Cover every output and constraint. Re-read the request and list every output
  it names — each file path, and each field, section, format, name, ordering,
  value range, or "all vs. one" rule stated about it. Confirm each one against
  your work (`ls`, `cat`); a single missing output or unmet constraint leaves
  the task unfinished.

- Verify the real behavior, not a proxy. Exercise the actual required operation
  end-to-end against adversarial and boundary inputs — the specific scenarios,
  parameter names, and edge cases the task describes — not a happy-path case you
  picked yourself. A check that only runs inputs you chose can pass while the
  behavior is still wrong.

- Make it reproducible from a clean state. Your work has to function for someone
  starting fresh, not only in the shell you built it in. A service must keep
  running on its own — a managed or persistent service, not a process tied to
  the shell that launched it (which dies when that shell exits). A script must
  run using only what is already installed; if you installed something ad hoc to
  make it work, it will fail elsewhere. Confirm it from a brand-new shell —
  restart the service, open a fresh session, re-run the script — not just where
  you built it.

- Stay within the task's scope. Modify only what the task asks you to produce or
  change, and don't reach into state it never named — don't rewrite shared
  history, migrate schemas, or regenerate or delete files you weren't asked to.
  This limits what you touch, not how much you do: doing the full work a task
  needs — installing packages, configuring and starting services, building out a
  complete setup — is expected, not a violation. Once your output is computed and
  cross-checked, record it and stop; don't launch another long run just to
  re-confirm a result you've already validated.
</verification_discipline>

<work_in_batches>
When iterating — building, testing, debugging, or reverse-engineering — do as
much as possible per command rather than one probe per turn. Script the whole
cycle (build, run, check) so it prints one consolidated result you can act on,
instead of running a command, reading a single value, and stopping. When
inspecting an unknown file, binary, or data structure, extract the specific
values you need in one pass rather than querying them one at a time. If one step
is unavoidably long (a large training, sampling, or build run), start it in the
background with a timeout and poll for completion, rather than blocking on a
single multi-minute command.
</work_in_batches>"""
"""Text appended to the assembled base system prompt."""


def _build_extra_middleware() -> list[FinalizeMiddleware | RambleMiddleware]:
    """Build fresh GLM-5.2 behavioral middleware instances for each agent stack.

    Used as the profile's ``extra_middleware`` factory so each assembled stack
    (main agent, general-purpose subagent, declarative subagents) gets its own
    instances rather than sharing per-run state across stacks.
    """
    return [FinalizeMiddleware(), RambleMiddleware()]


def register() -> None:
    """Register the built-in GLM-5.2 harness profile."""
    _register_harness_profile_impl(
        "fireworks:accounts/fireworks/models/glm-5p2",
        HarnessProfile(
            system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
            extra_middleware=_build_extra_middleware,
        ),
    )
