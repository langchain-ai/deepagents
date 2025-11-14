"""Shell tool middleware that survives HITL pauses."""

from langchain.agents.middleware.shell_tool import ShellToolMiddleware


class ResumableShellToolMiddleware(ShellToolMiddleware):
    """Shell middleware that recreates session resources after human interrupts.

    ``ShellToolMiddleware`` stores its session handle in middleware state using an
    ``UntrackedValue``. When a run pauses for human approval, that attribute is not
    checkpointed. Upon resuming, LangGraph restores the state without the shell
    resources, so the next tool execution fails with
    ``Shell session resources are unavailable``.

    This subclass lazily recreates the shell session the first time a resumed run
    touches the shell tool again and only performs shutdown when a session is
    actually active. This keeps behaviour identical for uninterrupted runs while
    allowing HITL pauses to succeed.
    """
    ...
