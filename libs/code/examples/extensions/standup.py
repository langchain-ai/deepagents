"""Reference extension: a slash command.

Registers `/standup`, which summarizes today's commits in the working directory.
Command handlers may be sync or async and receive a `CommandContext` carrying the
arguments, the session's working directory, and the runtime mode.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.extensions import CommandContext, ExtensionAPI


async def standup(ctx: CommandContext) -> str:
    """Summarize commits in the working directory since a given date.

    Args:
        ctx: Invocation context; the argument is a git `--since` expression.

    Returns:
        The commit list, or a message explaining why it is unavailable.
    """
    since = ctx.args or "yesterday"
    process = await asyncio.create_subprocess_exec(
        "git",
        "log",
        f"--since={since}",
        "--pretty=- %s",
        cwd=ctx.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        return f"git log failed: {stderr.decode().strip()}"
    return stdout.decode().strip() or f"No commits since {since}."


def extension(d: ExtensionAPI) -> None:
    """Register the `/standup` command.

    Args:
        d: The dcode extension API.
    """
    d.register_command("standup", standup, description="Summarize recent commits")
