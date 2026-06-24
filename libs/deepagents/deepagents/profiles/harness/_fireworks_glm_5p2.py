"""Built-in GLM-5.2 harness profile for Fireworks.

GLM-5.2 is used in the eval harness as a text-oriented model. The filesystem
tooling may encounter media files even for terminal-oriented tasks, so this
profile nudges the model toward inspecting media through the sandbox instead
of relying on direct visual input from the chat model.
"""

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
</media_file_handling>"""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in GLM-5.2 harness profile."""
    _register_harness_profile_impl(
        "fireworks:accounts/fireworks/models/glm-5p2",
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
