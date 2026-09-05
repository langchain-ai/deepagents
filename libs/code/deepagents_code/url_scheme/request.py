"""Parsing and validation for `dcode://` links.

A link can be published by any web page, chat message, or email, so this module
is the outer edge of the handler's trust boundary. It decides what a link is
allowed to say before any of it reaches a launch, and the rules are deliberately
narrow:

- One action, `open`. An unrecognized action is refused, not ignored.
- A closed parameter set. An unknown key is refused, so a link written for a
    newer dcode fails loudly here instead of being silently half-honored.
- One value per key. A repeated key is refused rather than resolved by a
    first- or last-wins rule the link's author cannot predict.
- No approval, sandbox, model, or hook parameters, and no way to express one. A
    link cannot weaken the approval posture of the session it opens, because
    `handler` passes the launch only the fields this module produces.
- No silent repair. Text carrying control characters or deceptive Unicode is
    refused instead of stripped, so the request the user reads in the
    confirmation is the request the session receives.

Query values are decoded as `application/x-www-form-urlencoded`, which is what
browsers and `URLSearchParams` produce and what `build_open_url` emits. One
consequence is worth knowing when writing a link by hand: `+` decodes to a
space, so a path or prompt containing a literal `+` has to be sent as `%2B`.
`_directory` says so when a decoded path holds a space and does not exist.

Everything here is shape validation. Text that survives it is still untrusted
prose: `handler` shows the whole request to the user and waits for an explicit
approval before anything runs.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit

URL_SCHEME: Final = "dcode"
"""Scheme dcode registers with the operating system.

A private scheme in the sense of RFC 7595 section 3.8 — not IANA-registered,
and named after the console script so a link reads as the command it runs.
"""

OPEN_ACTION: Final = "open"
"""The only action a link may request."""

MAX_URL_CHARS: Final = 8192
"""Longest link accepted, matching the URL length browsers reliably carry."""

MAX_PROMPT_CHARS: Final = 2000
"""Longest `prompt` accepted.

The bound is readability, not capacity: the user has to read the whole prompt in
the confirmation before approving it, and a wall of text is a prompt nobody
reads. Longer instructions belong in the session, typed by the user.
"""

_PARAMS: Final = frozenset({"dir", "thread", "agent", "prompt"})
"""Every parameter `open` accepts. Anything else is refused."""

_ACTION_RE: Final = re.compile(r"\A[a-z][a-z0-9-]*\Z")
"""Shape of an action token, checked before it is compared or reported."""

_AGENT_RE: Final = re.compile(r"\A[A-Za-z0-9_\- ]{1,64}\Z")
"""Shape of an agent name.

Mirrors the character class `_paths._validate_agent_name` enforces, with a
length bound added. Whether the agent exists is the launch's question, not this
module's.
"""

_PROMPT_ALLOWED_CONTROLS: Final = frozenset({"\t", "\n"})
"""Control characters a prompt may contain, so multi-line prompts stay legible."""

_C0_END: Final = 0x20
_C1_START: Final = 0x7F
_C1_END: Final = 0x9F


class UrlRequestError(ValueError):
    """A link is malformed, or asks for something a link may not ask for.

    The message reaches the user, so it says which part of the link was refused
    rather than only that the link was refused.
    """


@dataclass(frozen=True)
class OpenRequest:
    """A validated `dcode://open` request.

    Attributes:
        directory: Existing directory to start the session in, absolute and
            symlink-resolved so the confirmation names the real target.
        thread: Thread to resume, in canonical UUID form, or `None` for a new
            thread.
        agent: Agent to launch, or `None` for the configured default.
        prompt: Text to submit as the session's first message, or `None`.
    """

    directory: Path
    thread: str | None = None
    agent: str | None = None
    prompt: str | None = None


def parse_open_url(raw: str) -> OpenRequest:
    """Validate a `dcode://` link and return the request it makes.

    Args:
        raw: The link as the operating system delivered it.

    Returns:
        The validated request.

    Raises:
        UrlRequestError: The link is not a `dcode://open` link this version
            understands, or a parameter is malformed, unsupported, or unsafe.
    """
    if len(raw) > MAX_URL_CHARS:
        msg = f"Link is too long ({len(raw)} characters, limit {MAX_URL_CHARS})."
        raise UrlRequestError(msg)

    parts = urlsplit(raw)
    if parts.scheme.lower() != URL_SCHEME:
        msg = f"Not a {URL_SCHEME}:// link."
        raise UrlRequestError(msg)
    if parts.fragment:
        msg = f"{URL_SCHEME}:// links do not take a fragment."
        raise UrlRequestError(msg)

    _require_open_action(parts.netloc, parts.path)
    return _build_request(_query_params(parts.query))


def build_open_url(
    directory: Path | str,
    *,
    thread: str | None = None,
    agent: str | None = None,
    prompt: str | None = None,
) -> str:
    """Build a `dcode://open` link.

    The inverse of `parse_open_url`, used for the examples `dcode url status`
    prints and by callers that publish links. It encodes rather than validates:
    a link it returns is still checked in full when the handler receives it.

    Args:
        directory: Directory the link should open.
        thread: Thread to resume.
        agent: Agent to launch.
        prompt: First message to submit.

    Returns:
        The encoded link.
    """
    query: list[tuple[str, str]] = [("dir", str(directory))]
    query += [
        (key, value)
        for key, value in (("thread", thread), ("agent", agent), ("prompt", prompt))
        if value is not None
    ]
    return f"{URL_SCHEME}://{OPEN_ACTION}?{urlencode(query)}"


def _require_open_action(netloc: str, path: str) -> None:
    """Check that the link names the `open` action and nothing after it.

    Browsers hand over whatever the page wrote, so both spellings a link author
    may reach for are accepted: `dcode://open?...` carries the action in the
    authority component, and `dcode:open?...` carries it in the path.

    Args:
        netloc: Authority component of the link.
        path: Path component of the link.

    Raises:
        UrlRequestError: The action is missing, unrecognized, carries authority
            syntax such as userinfo or a port, or is followed by a path.
    """
    if netloc:
        action, trailing = netloc, path.strip("/")
    else:
        action, _, rest = path.lstrip("/").partition("/")
        trailing = rest.strip("/")

    if not action:
        msg = f"Link names no action; expected {URL_SCHEME}://{OPEN_ACTION}?..."
        raise UrlRequestError(msg)
    if not _ACTION_RE.match(action):
        msg = f"Link action is not a recognized action name: {action[:32]!r}."
        raise UrlRequestError(msg)
    if action != OPEN_ACTION:
        msg = (
            f"Unsupported action {action!r}. This version of dcode handles "
            f"{URL_SCHEME}://{OPEN_ACTION} only."
        )
        raise UrlRequestError(msg)
    if trailing:
        msg = f"{URL_SCHEME}://{OPEN_ACTION} takes no path; use query parameters."
        raise UrlRequestError(msg)


def _query_params(query: str) -> dict[str, str]:
    """Decode the query string into at most one value per supported key.

    Args:
        query: Raw query component of the link.

    Returns:
        Decoded parameters.

    Raises:
        UrlRequestError: The query is malformed, repeats a key, or names a key
            this version does not support.
    """
    if not query:
        return {}
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        msg = f"Link query could not be decoded: {exc}"
        raise UrlRequestError(msg) from exc

    params: dict[str, str] = {}
    for key, value in pairs:
        if key not in _PARAMS:
            supported = ", ".join(sorted(_PARAMS))
            msg = (
                f"Unsupported parameter {key[:32]!r}. Supported parameters: "
                f"{supported}. A link written for a newer dcode may need a "
                "dcode update."
            )
            raise UrlRequestError(msg)
        if key in params:
            msg = f"Parameter {key!r} appears more than once."
            raise UrlRequestError(msg)
        params[key] = value
    return params


def _build_request(params: dict[str, str]) -> OpenRequest:
    """Validate decoded parameters into an `OpenRequest`.

    Args:
        params: Decoded query parameters.

    Returns:
        The validated request.

    Raises:
        UrlRequestError: `dir` is missing or unusable, or another parameter is
            malformed.
    """
    raw_dir = params.get("dir")
    if not raw_dir:
        msg = (
            "Link is missing the 'dir' parameter, which says where to open the session."
        )
        raise UrlRequestError(msg)
    return OpenRequest(
        directory=_directory(raw_dir),
        thread=_thread(params.get("thread")),
        agent=_agent(params.get("agent")),
        prompt=_prompt(params.get("prompt")),
    )


def _directory(value: str) -> Path:
    """Resolve the `dir` parameter to an existing directory.

    A relative path is refused rather than joined onto something: the handler
    runs in whatever directory the desktop launcher happened to choose, which
    has nothing to do with the link's author, so a relative path has no meaning
    here.

    Args:
        value: Raw `dir` value.

    Returns:
        The absolute, symlink-resolved directory.

    Raises:
        UrlRequestError: The path is not absolute, does not exist, is
            unreadable, or is not a directory.
    """
    _reject_control_chars(value, field="dir", allowed=frozenset())
    try:
        expanded = Path(value).expanduser()
    except (RuntimeError, ValueError) as exc:
        # `expanduser` raises when `~` cannot be resolved to a home directory.
        msg = f"Link directory could not be read as a path: {value[:120]!r}"
        raise UrlRequestError(msg) from exc

    if not expanded.is_absolute():
        msg = (
            "Link directory must be an absolute path (or start with '~'): "
            f"{value[:120]!r}"
        )
        raise UrlRequestError(msg)

    try:
        resolved = expanded.resolve(strict=True)
        is_dir = resolved.is_dir()
    except OSError as exc:
        msg = f"Link directory does not exist or is unreadable: {expanded}"
        if " " in value:
            # The likeliest cause of a path that gained a space: a query decodes
            # `+` as a space, so a literal `+` has to be written as `%2B`.
            msg += " (a '+' in a path must be encoded as %2B)"
        raise UrlRequestError(msg) from exc
    if not is_dir:
        msg = f"Link directory is not a directory: {resolved}"
        raise UrlRequestError(msg)
    return resolved


def _thread(value: str | None) -> str | None:
    """Validate the `thread` parameter as a thread identifier.

    Thread ids are UUIDs (`sessions.generate_thread_id` mints UUID7), so the
    value is parsed as one and returned in canonical form. That also keeps the
    resume sentinels the CLI understands out of a link's reach: `-r` reads
    `__MOST_RECENT__` as "resume whatever I last worked on", which is not a
    thread a link's author is in a position to name.

    Args:
        value: Raw `thread` value, or `None`.

    Returns:
        The canonical UUID string, or `None`.

    Raises:
        UrlRequestError: The value is not a UUID.
    """
    if value is None:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        msg = f"Link thread id is not a valid thread id: {value[:64]!r}"
        raise UrlRequestError(msg) from exc


def _agent(value: str | None) -> str | None:
    """Validate the `agent` parameter as an agent name.

    Args:
        value: Raw `agent` value, or `None`.

    Returns:
        The agent name, or `None`.

    Raises:
        UrlRequestError: The name has an unusable shape, or is reserved for
            dcode's own state.
    """
    if value is None:
        return None
    if not _AGENT_RE.match(value):
        msg = (
            f"Link agent name is not a usable agent name: {value[:64]!r}. Agent "
            "names hold letters, numbers, hyphens, underscores, and spaces."
        )
        raise UrlRequestError(msg)
    from deepagents_code._reserved_names import is_reserved_agent_dir_name

    if is_reserved_agent_dir_name(value):
        msg = f"Link agent name {value!r} is reserved for dcode's own state."
        raise UrlRequestError(msg)
    return value


def _prompt(value: str | None) -> str | None:
    """Validate the `prompt` parameter as text safe to display and to submit.

    The prompt is the one parameter that is prose rather than a shape, and the
    confirmation shows it verbatim. Anything that could make the rendered text
    disagree with the submitted text — escape sequences, bidi overrides,
    invisible code points — is refused rather than stripped, so the two cannot
    diverge.

    Args:
        value: Raw `prompt` value, or `None`.

    Returns:
        The prompt text, or `None` when absent or blank.

    Raises:
        UrlRequestError: The prompt is too long, or carries control characters
            or deceptive Unicode.
    """
    if value is None or not value.strip():
        return None
    if len(value) > MAX_PROMPT_CHARS:
        msg = (
            f"Link prompt is too long ({len(value)} characters, limit "
            f"{MAX_PROMPT_CHARS}). Open the session and type the rest."
        )
        raise UrlRequestError(msg)
    _reject_control_chars(value, field="prompt", allowed=_PROMPT_ALLOWED_CONTROLS)

    from deepagents_code.unicode_security import (
        detect_dangerous_unicode,
        summarize_issues,
    )

    issues = detect_dangerous_unicode(value)
    if issues:
        msg = (
            "Link prompt contains hidden or direction-changing characters "
            f"({summarize_issues(issues)}), so what you would read is not "
            "necessarily what would be sent."
        )
        raise UrlRequestError(msg)
    return value


def _reject_control_chars(value: str, *, field: str, allowed: frozenset[str]) -> None:
    """Refuse text carrying control characters.

    Args:
        value: Text to check.
        field: Parameter name, named in the message.
        allowed: Control characters this field may contain.

    Raises:
        UrlRequestError: The text holds a C0 or C1 control character outside
            `allowed`.
    """
    for char in value:
        code = ord(char)
        if char in allowed or not (code < _C0_END or _C1_START <= code <= _C1_END):
            continue
        msg = (
            f"Link {field} contains a control character (U+{code:04X}), which a "
            "link may not carry."
        )
        raise UrlRequestError(msg)
