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
</media_file_handling>

<verification_discipline>
Before treating a task as complete, self-audit your work:
- Deliverables: re-read the request and list every output it names — each file path,
  and each field, section, or format required inside it. Confirm each exists with the
  required content (`ls`, `cat`), not just the main one; a single missing or empty
  output leaves the task unfinished.
- Persistence: a service the task needs (web server, daemon, `sshd`, database) must
  keep running independently of the shell that started it — a process launched in a
  command shell may die when that command returns. Start it so it survives (the
  system's init/service manager, or a disowned `nohup … &`/`setsid` process) and
  confirm it is actually serving by connecting from a new shell, not by checking the
  PID you just launched.
- Honest checks: exercise the real required behavior, not a proxy (do not grep for a
  substring in place of running the code). If a check you wrote fails, fix the
  implementation — never weaken, narrow, or replace the check to make it pass.
</verification_discipline>"""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in GLM-5.2 harness profile."""
    _register_harness_profile_impl(
        "fireworks:accounts/fireworks/models/glm-5p2",
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
