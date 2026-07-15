"""Integration coverage for server-side goal criteria on the main graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import ToolMessage

from deepagents_code._testing_models import DCA_TEST_GOAL_CRITERIA_MARKER

if TYPE_CHECKING:
    from pathlib import Path


def _write_model_config(home_dir: Path) -> None:
    """Configure the deterministic criteria integration model."""
    config_dir = home_dir / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[models.providers.itest]
class_path = "deepagents_code._testing_models:GoalCriteriaIntegrationChatModel"
models = ["goal-criteria"]
""".strip()
        + "\n"
    )


@pytest.mark.timeout(180)
async def test_goal_criteria_runs_inside_main_server_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main graph should persist a proposal without polluting parent history."""
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (project_dir / "context.txt").write_text("server-only context\n")
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)
    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.client.launch.server_manager import server_session
    from deepagents_code.config import build_stream_config
    from deepagents_code.sessions import generate_thread_id

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    model_config.clear_caches()
    thread_id = generate_thread_id()
    saw_repository_read = False
    parent_messages = [
        {"role": "user", "content": "The relevant code is under src/auth/."},
        {"role": "assistant", "content": "I found the existing login flow."},
    ]

    try:
        async with server_session(
            assistant_id="itest-goal-criteria",
            model_name="itest:goal-criteria",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            config = build_stream_config(thread_id, "itest-goal-criteria")
            async for namespace, mode, data in agent.astream(
                {
                    "messages": parent_messages,
                    "goal_criteria_request": {
                        "request_id": "request-1",
                        "kind": "create",
                        "objective": "verify server-side criteria generation",
                        "feedback": f"{DCA_TEST_GOAL_CRITERIA_MARKER}/context.txt",
                    },
                },
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
                context={"thread_id": thread_id, "auto_approve": True},
            ):
                if (
                    namespace
                    and mode == "messages"
                    and isinstance(data, tuple)
                    and isinstance(data[0], ToolMessage)
                    and data[0].name == "read_file"
                ):
                    saw_repository_read = True

            state = await agent.aget_state(dict(config))
            assert state is not None
            assert state.values["_pending_goal_objective"] == (
                "verify server-side criteria generation"
            )
            assert state.values["_pending_goal_rubric"] == (
                "- server repository context is available"
            )
            assert state.values["_pending_goal_kind"] == "create"
            messages = state.values.get("messages", [])
            assert len(messages) == 2
            assert [(message["type"], message["content"]) for message in messages] == [
                ("human", parent_messages[0]["content"]),
                ("ai", parent_messages[1]["content"]),
            ]
            assert state.values.get("goal_criteria_request") is None
            assert saw_repository_read
    finally:
        model_config.clear_caches()
