from pathlib import Path

import pytest
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Markdown

from deepagents_code.tui.widgets.tool_renderers import get_renderer
from deepagents_code.tui.widgets.tool_widgets import (
    _MAX_LINES,
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
)

_CREDENTIAL_NOTICE_FRAGMENT = "may contain credentials"


def _widget_texts(widgets: list[Widget]) -> list[str]:
    """Return the plain text rendered by each widget, ignoring styles."""
    texts: list[str] = []
    for widget in widgets:
        rendered = widget.render()
        texts.append(rendered.plain if isinstance(rendered, Content) else str(rendered))
    return texts


def test_write_renderer_formats_non_string_content() -> None:
    widget_class, data = get_renderer("write_file").get_approval_widget(
        {"file_path": "data.json", "content": {"a": "b"}}
    )

    assert widget_class is WriteFileApprovalWidget
    assert data["content"] == '{\n  "a": "b"\n}'


def test_write_renderer_falls_back_to_str_for_unserializable_content() -> None:
    widget_class, data = get_renderer("write_file").get_approval_widget(
        {"file_path": "data.txt", "content": {1, 2, 3}}
    )

    assert widget_class is WriteFileApprovalWidget
    assert data["content"] == str({1, 2, 3})


def test_write_widget_formats_non_string_content() -> None:
    widgets = list(
        WriteFileApprovalWidget(
            {"file_path": "data.json", "content": {"a": "b"}}
        ).compose()
    )

    assert len(widgets) == 3


@pytest.mark.parametrize(
    "file_path",
    [".env", "/home/user/project/.env", "config/.env.local"],
)
def test_write_widget_redacts_credential_file_content(file_path: str) -> None:
    widgets = list(
        WriteFileApprovalWidget(
            {
                "file_path": file_path,
                "content": "SECRET_KEY=supersecret",
                "file_extension": "text",
            }
        ).compose()
    )

    assert any(_CREDENTIAL_NOTICE_FRAGMENT in text for text in _widget_texts(widgets))
    # Content is rendered via `Markdown`, whose `render()` yields the widget
    # name rather than its source — so `_widget_texts` cannot see a leak there
    # and a substring check alone would pass even if the secret slipped
    # through. Guard structurally: the credential branch must emit no
    # `Markdown` child at all.
    assert not any(isinstance(widget, Markdown) for widget in widgets)


def test_write_widget_renders_regular_file_via_markdown() -> None:
    """Positive control for the credential redaction test.

    A non-credential file must use the `Markdown` branch; without this, the
    "no `Markdown` child" assertion above could pass vacuously if the widget
    stopped using `Markdown` for everything.
    """
    widgets = list(
        WriteFileApprovalWidget(
            {
                "file_path": "main.py",
                "content": "print('hi')",
                "file_extension": "python",
            }
        ).compose()
    )

    assert any(isinstance(widget, Markdown) for widget in widgets)


def test_write_widget_redacts_large_credential_file() -> None:
    """A credential file longer than the truncation limit must not leak.

    The redaction check must short-circuit before the truncation branch,
    which would otherwise render the first `_MAX_LINES` lines via `Markdown`.
    """
    content = "\n".join(f"SECRET_{i}=value{i}" for i in range(_MAX_LINES + 20))
    widgets = list(
        WriteFileApprovalWidget(
            {"file_path": ".env", "content": content, "file_extension": "text"}
        ).compose()
    )

    assert any(_CREDENTIAL_NOTICE_FRAGMENT in text for text in _widget_texts(widgets))
    assert not any(isinstance(widget, Markdown) for widget in widgets)


def test_edit_renderer_formats_non_string_content() -> None:
    widget_class, data = get_renderer("edit_file").get_approval_widget(
        {
            "file_path": "data.json",
            "old_string": {"a": "b"},
            "new_string": {"a": "c"},
        }
    )

    assert widget_class is EditFileApprovalWidget
    assert data["old_string"] == '{\n  "a": "b"\n}'
    assert data["new_string"] == '{\n  "a": "c"\n}'
    assert '-  "a": "b"' in data["diff_lines"]
    assert '+  "a": "c"' in data["diff_lines"]


def test_edit_widget_formats_non_string_content() -> None:
    widgets = list(
        EditFileApprovalWidget(
            {
                "file_path": "data.json",
                "old_string": {"a": "b"},
                "new_string": {"a": "c"},
            }
        ).compose()
    )

    assert widgets


def test_edit_widget_redacts_credential_file_diff() -> None:
    widgets = list(
        EditFileApprovalWidget(
            {
                "file_path": "config/.env.local",
                "diff_lines": ["+API_TOKEN=leaked"],
                "old_string": "",
                "new_string": "API_TOKEN=leaked",
            }
        ).compose()
    )

    texts = _widget_texts(widgets)
    assert any(_CREDENTIAL_NOTICE_FRAGMENT in text for text in texts)
    assert all("leaked" not in text for text in texts)


def test_delete_renderer_shows_removed_file_diff(tmp_path: Path) -> None:
    target = tmp_path / "old.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    widget_class, data = get_renderer("delete").get_approval_widget(
        {"file_path": str(target)}
    )

    assert widget_class is EditFileApprovalWidget
    assert data["file_path"] == "old.txt"
    assert "-alpha" in data["diff_lines"]
    assert "-beta" in data["diff_lines"]


def test_delete_widget_redacts_credential_file(tmp_path: Path) -> None:
    """Deleting a credential file must not show its removed contents.

    Delete routes through `EditFileApprovalWidget` with the display-formatted
    path, so this pins that the sensitivity check still fires on that path.
    """
    target = tmp_path / ".env"
    target.write_text("API_KEY=supersecret\n", encoding="utf-8")

    widget_class, data = get_renderer("delete").get_approval_widget(
        {"file_path": str(target)}
    )
    widgets = list(widget_class(data).compose())

    texts = _widget_texts(widgets)
    assert any(_CREDENTIAL_NOTICE_FRAGMENT in text for text in texts)
    assert all("supersecret" not in text for text in texts)


def test_delete_renderer_flags_directories_without_diff(tmp_path: Path) -> None:
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "child.txt").write_text("data\n", encoding="utf-8")

    widget_class, data = get_renderer("delete").get_approval_widget(
        {"file_path": str(target)}
    )

    assert widget_class is GenericApprovalWidget
    assert data["file_path"] == "subdir"
    assert "Contents: directory or unreadable file" in data["details"]


def test_delete_renderer_surfaces_unresolvable_path_error() -> None:
    """An empty path yields a resolution error shown in the approval widget."""
    widget_class, data = get_renderer("delete").get_approval_widget({"file_path": ""})

    assert widget_class is GenericApprovalWidget
    assert data["error"] == "Unable to resolve file path."
