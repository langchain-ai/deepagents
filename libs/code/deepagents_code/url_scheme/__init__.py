"""`dcode://` URL-scheme support: registration, parsing, and handling.

A page can offer to open a directory or resume a thread in dcode by linking to
`dcode://open?dir=/path/to/project`, the way a page offers to open an editor or
a chat client. The browser asks whether to hand the link to dcode; dcode then
asks whether to honor this particular request.

The feature is three pieces, each with its own module docstring:

- `request` validates a link. It is the trust boundary: one action, a closed
    parameter set, and no parameter that could weaken the session a link opens.
- `registration` claims and releases the scheme with the operating system, via
    one backend per desktop stack (`_macos`, `_linux`, `_windows`).
- `handler` runs in the terminal the desktop opened: it shows the request in
    full, waits for an explicit approval, and then launches the session.

Nothing here runs unless the user asks for it twice — once by registering the
scheme, and again for each link they approve.
"""

from __future__ import annotations

from deepagents_code.url_scheme.handler import (
    EXIT_DECLINED,
    EXIT_REFUSED,
    open_from_url,
)
from deepagents_code.url_scheme.registration import (
    HandlerStatus,
    RegistrationError,
    TerminalChoice,
    handler_status,
    install_handler,
    resolve_launcher,
    uninstall_handler,
)
from deepagents_code.url_scheme.request import (
    MAX_PROMPT_CHARS,
    OPEN_ACTION,
    URL_SCHEME,
    OpenRequest,
    UrlRequestError,
    build_open_url,
    parse_open_url,
)

__all__ = [
    "EXIT_DECLINED",
    "EXIT_REFUSED",
    "MAX_PROMPT_CHARS",
    "OPEN_ACTION",
    "URL_SCHEME",
    "HandlerStatus",
    "OpenRequest",
    "RegistrationError",
    "TerminalChoice",
    "UrlRequestError",
    "build_open_url",
    "handler_status",
    "install_handler",
    "open_from_url",
    "parse_open_url",
    "resolve_launcher",
    "uninstall_handler",
]
