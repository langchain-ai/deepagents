"""Effective interpreter permissions in durable workspace bindings."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from deepagents_code._server_config import ServerConfig
from deepagents_code.configuration.interpreter import InterpreterConfig
from deepagents_code.configuration.providers import DefaultProvider, TomlFileProvider
from deepagents_code.configuration.resolver import (
    MANAGED_RANK,
    USER_RANK,
    ConfigResolver,
)
from deepagents_code.workspace import WorkspaceConflictError, bind_thread_workspace

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def policy_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def install(text: str = "", *, managed: bool = False) -> ConfigResolver:
        path = tmp_path / "policy.toml"
        path.write_text(text)
        resolver = ConfigResolver(
            (
                TomlFileProvider(
                    "test policy", path, MANAGED_RANK if managed else USER_RANK
                ),
                DefaultProvider(),
            )
        )
        monkeypatch.setattr(
            "deepagents_code.configuration.interpreter.get_config_resolver",
            lambda: resolver,
        )
        return resolver

    monkeypatch.setenv("DEEPAGENTS_CODE_SERVER_DB_PATH", str(tmp_path / "sessions.db"))
    return install


@pytest.mark.parametrize("explicit_first", [True, False])
async def test_implicit_and_explicit_safe_resume(
    tmp_path, policy_resolver, explicit_first
):
    policy_resolver()
    omitted = ServerConfig(enable_interpreter=True)
    explicit = replace(omitted, interpreter_ptc="safe")
    first, second = (explicit, omitted) if explicit_first else (omitted, explicit)
    bound = await bind_thread_workspace(
        "thread", str(tmp_path), first.to_workspace_payload()
    )
    assert (
        await bind_thread_workspace(
            "thread", str(tmp_path), second.to_workspace_payload()
        )
        == bound
    )


async def test_existing_explicit_safe_binding_resumes(tmp_path, policy_resolver):
    policy_resolver()
    config = ServerConfig(enable_interpreter=True, interpreter_ptc="safe")
    historical = config.to_workspace_payload()
    await bind_thread_workspace("thread", str(tmp_path), historical)
    await bind_thread_workspace(
        "thread",
        str(tmp_path),
        replace(config, interpreter_ptc=None).to_workspace_payload(),
    )


@pytest.mark.parametrize("historical", [None, ["task"], "all"])
async def test_unknown_or_different_historical_permissions_refused(
    tmp_path, policy_resolver, historical
):
    policy_resolver()
    config = ServerConfig(enable_interpreter=True)
    old = config.to_workspace_payload() | {"interpreter_ptc": historical}
    await bind_thread_workspace("thread", str(tmp_path), old)
    with pytest.raises(WorkspaceConflictError):
        await bind_thread_workspace(
            "thread", str(tmp_path), config.to_workspace_payload()
        )


async def test_configured_allowlist_and_drift(tmp_path, policy_resolver):
    policy_resolver('[interpreter]\nptc = ["task"]\n')
    config = ServerConfig(enable_interpreter=True)
    await bind_thread_workspace("thread", str(tmp_path), config.to_workspace_payload())
    await bind_thread_workspace(
        "thread",
        str(tmp_path),
        replace(config, interpreter_ptc=["task"]).to_workspace_payload(),
    )
    with pytest.raises(WorkspaceConflictError):
        await bind_thread_workspace(
            "thread",
            str(tmp_path),
            replace(config, interpreter_ptc="safe").to_workspace_payload(),
        )
    policy_resolver()
    with pytest.raises(WorkspaceConflictError):
        await bind_thread_workspace(
            "thread", str(tmp_path), config.to_workspace_payload()
        )


async def test_managed_permissions_override_invocation(tmp_path, policy_resolver):
    resolver = policy_resolver(
        '[interpreter]\nptc = "safe"\nptc_acknowledge_unsafe = false\n', managed=True
    )
    config = ServerConfig(
        enable_interpreter=True,
        interpreter_ptc="all",
        interpreter_ptc_acknowledge_unsafe=True,
    )
    runtime = InterpreterConfig.from_resolver(
        resolver, ptc=config.interpreter_ptc, ptc_acknowledge_unsafe=True
    )
    assert runtime.ptc == config.to_workspace_payload()["interpreter_ptc"] == "safe"
    assert runtime.ptc_acknowledge_unsafe is False
    await bind_thread_workspace("thread", str(tmp_path), config.to_workspace_payload())
    await bind_thread_workspace(
        "thread",
        str(tmp_path),
        replace(
            config, interpreter_ptc=None, interpreter_ptc_acknowledge_unsafe=False
        ).to_workspace_payload(),
    )


async def test_acknowledgement_and_other_policy_drift_refused(
    tmp_path, policy_resolver
):
    policy_resolver()
    config = ServerConfig(
        enable_interpreter=True,
        interpreter_ptc="all",
        interpreter_ptc_acknowledge_unsafe=True,
    )
    await bind_thread_workspace("thread", str(tmp_path), config.to_workspace_payload())
    for changed in (
        replace(config, interpreter_ptc_acknowledge_unsafe=False),
        replace(config, auto_approve=True),
    ):
        with pytest.raises(WorkspaceConflictError):
            await bind_thread_workspace(
                "thread", str(tmp_path), changed.to_workspace_payload()
            )


async def test_runtime_validation_uses_effective_policy(
    tmp_path, policy_resolver, monkeypatch
):
    from deepagents_code import server_graph
    from deepagents_code.workspace import resolve_workspace

    policy_resolver()
    explicit = ServerConfig(enable_interpreter=True, interpreter_ptc="safe")
    binding = resolve_workspace(str(tmp_path), explicit.to_workspace_payload())
    omitted = replace(explicit, interpreter_ptc=None)
    monkeypatch.setattr(ServerConfig, "from_env", lambda: omitted)
    resolved = await server_graph._resolve_bound_workspace_config(binding)
    assert resolved.to_workspace_payload()["interpreter_ptc"] == "safe"
    policy_resolver('[interpreter]\nptc = ["task"]\n')
    with pytest.raises(WorkspaceConflictError):
        await server_graph._resolve_bound_workspace_config(binding)


def test_unsafe_all_still_requires_acknowledgement(policy_resolver):
    from deepagents_code.agent import _resolve_ptc_option

    policy_resolver('[interpreter]\nptc = "all"\n')
    runtime = InterpreterConfig.from_resolver()
    with pytest.raises(ValueError, match="acknowledge"):
        _resolve_ptc_option(
            runtime.ptc,
            tools=[],
            acknowledge_unsafe=runtime.ptc_acknowledge_unsafe,
            auto_approve=False,
        )
    acknowledged = InterpreterConfig.from_resolver(ptc_acknowledge_unsafe=True)
    assert (
        _resolve_ptc_option(
            acknowledged.ptc,
            tools=[],
            acknowledge_unsafe=acknowledged.ptc_acknowledge_unsafe,
            auto_approve=False,
        )
        == []
    )


async def test_target_workspace_environment_keeps_runtime_policy(
    tmp_path, policy_resolver, monkeypatch
):
    from deepagents_code import server_graph
    from deepagents_code.config import _preview_dotenv_environ, use_environment
    from deepagents_code.workspace import resolve_workspace

    policy_resolver('[interpreter]\nptc = ["task"]\n')
    target = tmp_path / "target"
    target.mkdir()
    (target / ".env").write_text("WORKSPACE_ONLY=target\n")
    launch = ServerConfig(cwd=str(tmp_path), enable_interpreter=True)
    identity = resolve_workspace(str(target))
    scoped = launch.resolve_workspace(identity.cwd, identity.project_root)
    binding = resolve_workspace(str(target), scoped.to_workspace_payload())
    monkeypatch.setattr(ServerConfig, "from_env", lambda: launch)
    resolved = await server_graph._resolve_bound_workspace_config(binding)
    with use_environment(_preview_dotenv_environ(start_path=target)):
        runtime = InterpreterConfig.from_resolver(ptc=resolved.interpreter_ptc)
    assert runtime.ptc == binding.workspace_config()["interpreter_ptc"] == ["task"]
