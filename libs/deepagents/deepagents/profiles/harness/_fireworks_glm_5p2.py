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
Before treating a task as complete, re-read the request and list every output it
names — each file path, and each field, section, or format required inside it.
Confirm each exists with the required content (`ls`, `cat`), not just the main one;
a single missing or empty output leaves the task unfinished.
</verification_discipline>"""
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
