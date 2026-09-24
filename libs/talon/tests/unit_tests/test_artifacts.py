from pathlib import Path

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_talon.runtime import DeepAgentRuntime


@pytest.mark.parametrize("explicit_directory", [False, True])
@pytest.mark.parametrize("existing_artifacts", [False, True])
def test_history_offloads_inside_assistant_home(
    tmp_path: Path, *, explicit_directory: bool, existing_artifacts: bool
) -> None:
    home = tmp_path / "configured" / "test"
    assistant = tmp_path / "explicit" if explicit_directory else home
    if existing_artifacts:
        artifacts = assistant / "artifacts"
        artifacts.mkdir(parents=True)
        assistant.chmod(0o755)
        artifacts.chmod(0o755)
    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=assistant if explicit_directory else None,
        env={
            "DEEPAGENTS_TALON_HOME": str(home.parent),
            "DEEPAGENTS_TALON_ASSISTANT_ID": "test",
        },
    )
    middleware = SummarizationMiddleware(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="summary")]),
        backend=runtime.backend,
    )
    path = middleware._offload_to_backend(
        runtime.backend, [HumanMessage(content="Remember this conversation")], "session"
    )
    expected = assistant / "artifacts" / "conversation_history" / "session.md"
    assert path == str(expected)
    assert "Remember this conversation" in expected.read_text()
    assert (assistant / "artifacts").stat().st_mode & 0o777 == 0o700


def test_custom_backend_is_preserved(tmp_path: Path) -> None:
    backend = StateBackend()
    runtime = DeepAgentRuntime(model="test:model", backend=backend, assistant_dir=tmp_path)
    assert runtime.backend is backend
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("existing_target", [False, True])
def test_artifact_symlink_does_not_modify_target(tmp_path: Path, *, existing_target: bool) -> None:
    assistant = tmp_path / "assistant"
    assistant.mkdir()
    target = tmp_path / "shared"
    if existing_target:
        target.mkdir()
        target.chmod(0o755)
        (target / "data.txt").write_text("shared data")
    (assistant / "artifacts").symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError, match="artifacts"):
        DeepAgentRuntime(model="test:model", assistant_dir=assistant, env={})

    if existing_target:
        assert target.stat().st_mode & 0o777 == 0o755
        assert (target / "data.txt").read_text() == "shared data"
    else:
        assert not target.exists()
