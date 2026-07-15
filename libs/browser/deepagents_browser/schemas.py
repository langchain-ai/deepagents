"""Strict input schemas for browser tools."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=2_000)]
Reference = Annotated[
    str,
    StringConstraints(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]


class _StrictSchema(BaseModel):
    """Base schema that rejects unknown input fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class NavigateInput(_StrictSchema):
    """Input for navigating the active tab."""

    url: Annotated[str, StringConstraints(min_length=1, max_length=8_192)]
    page_ref: Reference | None = None


class SnapshotInput(_StrictSchema):
    """Input for capturing an actionable page snapshot."""

    page_ref: Reference | None = None


class ClickAction(_StrictSchema):
    """Click an element identified by an opaque snapshot reference."""

    kind: Literal["click"]
    ref: Reference


class TypeAction(_StrictSchema):
    """Replace an editable element's value with bounded text."""

    kind: Literal["type"]
    ref: Reference
    text: ShortText


class PressAction(_StrictSchema):
    """Press one allowlisted keyboard key on an element."""

    kind: Literal["press"]
    ref: Reference
    key: Literal[
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "Backspace",
        "Delete",
        "End",
        "Enter",
        "Escape",
        "Home",
        "PageDown",
        "PageUp",
        "Space",
        "Tab",
    ]


class SelectAction(_StrictSchema):
    """Select one option by bounded value."""

    kind: Literal["select"]
    ref: Reference
    value: ShortText


BrowserAction = Annotated[
    ClickAction | TypeAction | PressAction | SelectAction,
    Field(discriminator="kind"),
]


class ActInput(_StrictSchema):
    """Input for one allowlisted browser action."""

    action: BrowserAction


class ScreenshotInput(_StrictSchema):
    """Input for taking a bounded, fixed-viewport PNG screenshot."""

    page_ref: Reference | None = None


class TabsInput(_StrictSchema):
    """Input for listing or manipulating bounded browser tabs."""

    operation: Literal["list", "new", "select", "close"] = "list"
    page_ref: Reference | None = None
