"""Tests for the built-in extension provenance display."""

from deepagents_code.app import DeepAgentsApp


def test_extension_display_escapes_endpoint_metadata() -> None:
    """Paths and errors cannot inject markdown structure into the transcript."""
    rendered = DeepAgentsApp._render_extensions(
        {
            "registrations": [
                {
                    "kind": "tool",
                    "name": "name|forged",
                    "source": {
                        "scope": "project",
                        "path": "<b>/tmp/[extension]</b>",
                    },
                }
            ],
            "errors": ["bad\n# injected"],
            "restart_required": True,
        }
    )

    assert "name\\|forged" in rendered
    assert "\\<b\\>" in rendered
    assert "bad # injected" in rendered
    assert "Run `/restart`" in rendered
