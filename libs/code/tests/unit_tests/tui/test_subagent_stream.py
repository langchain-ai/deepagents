"""Unit tests for the subagent custom-stream boundary filter.

`_is_renderable_subagent_event` is the trust boundary between the agent's
`custom` stream and the live panel: only well-formed subagent events from the
main agent's namespace are forwarded.
"""

from __future__ import annotations

from deepagents_code.tui.textual_adapter import _is_renderable_subagent_event


def _event(**overrides: object) -> dict:
    event = {"type": "subagent", "id": "a"}
    event.update(overrides)
    return event


def test_accepts_well_formed_main_agent_event() -> None:
    assert _is_renderable_subagent_event(_event(), is_main_agent=True) is True


def test_rejects_nested_namespace() -> None:
    # Subagent-to-subagent emissions (non-empty namespace) are ignored.
    assert _is_renderable_subagent_event(_event(), is_main_agent=False) is False


def test_rejects_non_subagent_custom_payload() -> None:
    # Unrelated custom events (some other producer) must not reach the panel.
    assert (
        _is_renderable_subagent_event({"type": "progress"}, is_main_agent=True) is False
    )


def test_rejects_payload_without_type() -> None:
    assert _is_renderable_subagent_event({"id": "a"}, is_main_agent=True) is False


def test_rejects_non_dict_payload() -> None:
    assert _is_renderable_subagent_event("nope", is_main_agent=True) is False
    assert _is_renderable_subagent_event(None, is_main_agent=True) is False
