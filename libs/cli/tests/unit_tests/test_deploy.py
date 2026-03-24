"""Tests for the deploy module: config parsing, bundling, and langgraph.json generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.config import (
    BackendConfig,
    DeployConfig,
    MemoryConfig,
    NamespaceConfig,
    SandboxConfig,
    SkillsConfig,
    ToolsConfig,
)


# ------------------------------------------------------------------
# DeployConfig defaults
# ------------------------------------------------------------------


class TestDeployConfigDefaults:
    def test_default_config(self) -> None:
        config = DeployConfig()
        assert config.agent == "agent"
        assert config.model == "anthropic:claude-sonnet-4-6"
        assert config.backend.type == "store"
        assert config.backend.namespace.scope == "assistant"
        assert config.backend.namespace.prefix == "filesystem"
        assert config.sandbox is not None
        assert config.sandbox.provider == "langsmith"
        assert config.sandbox.scope == "thread"
        assert config.memory.scope == "assistant"
        assert config.tools.shell is True
        assert config.tools.web_search is True
        assert config.python_version == "3.12"

    def test_empty_dict_gives_defaults(self) -> None:
        config = DeployConfig.from_dict({})
        assert config.agent == "agent"
        assert config.model == "anthropic:claude-sonnet-4-6"
        assert config.sandbox is not None
        assert config.sandbox.provider == "langsmith"


# ------------------------------------------------------------------
# DeployConfig from_dict
# ------------------------------------------------------------------


class TestDeployConfigFromDict:
    def test_full_config(self) -> None:
        data = {
            "agent": "researcher",
            "description": "A research agent",
            "model": "openai:gpt-5",
            "model_params": {"temperature": 0.7},
            "prompt": "You are a research assistant.",
            "memory": {
                "scope": "user",
                "sources": [".deepagents/AGENTS.md", "custom/AGENTS.md"],
            },
            "skills": {
                "sources": [".deepagents/skills", "extra/skills"],
            },
            "tools": {
                "shell": False,
                "web_search": True,
                "fetch_url": False,
                "http_request": True,
                "mcp": ".mcp.json",
                "custom": "./my_tools.py:tools",
            },
            "backend": {
                "type": "store",
                "namespace": {"scope": "user", "prefix": "files"},
            },
            "sandbox": {
                "provider": "modal",
                "scope": "assistant",
                "template": "my-template",
                "image": "my-image:latest",
                "setup_script": "./setup.sh",
            },
            "env": ".env.production",
            "python_version": "3.13",
        }
        config = DeployConfig.from_dict(data)
        assert config.agent == "researcher"
        assert config.description == "A research agent"
        assert config.model == "openai:gpt-5"
        assert config.model_params == {"temperature": 0.7}
        assert config.prompt == "You are a research assistant."
        assert config.memory.scope == "user"
        assert len(config.memory.sources) == 2
        assert len(config.skills.sources) == 2
        assert config.tools.shell is False
        assert config.tools.custom == "./my_tools.py:tools"
        assert config.tools.mcp == ".mcp.json"
        assert config.backend.namespace.scope == "user"
        assert config.backend.namespace.prefix == "files"
        assert config.sandbox is not None
        assert config.sandbox.provider == "modal"
        assert config.sandbox.scope == "assistant"
        assert config.sandbox.template == "my-template"
        assert config.env == ".env.production"
        assert config.python_version == "3.13"

    def test_sandbox_disabled(self) -> None:
        config = DeployConfig.from_dict({"sandbox": False})
        assert config.sandbox is None

    def test_sandbox_none_gives_defaults(self) -> None:
        config = DeployConfig.from_dict({"sandbox": None})
        assert config.sandbox is not None
        assert config.sandbox.provider == "langsmith"

    def test_shell_allow_list_string(self) -> None:
        config = DeployConfig.from_dict({
            "tools": {"shell_allow_list": "git, python, pip"},
        })
        assert config.tools.shell_allow_list == ["git", "python", "pip"]

    def test_shell_allow_list_as_list(self) -> None:
        config = DeployConfig.from_dict({
            "tools": {"shell_allow_list": ["git", "python"]},
        })
        assert config.tools.shell_allow_list == ["git", "python"]


# ------------------------------------------------------------------
# Config file loading
# ------------------------------------------------------------------


class TestDeployConfigLoad:
    def test_load_from_file(self, tmp_path: Path) -> None:
        config_data = {
            "agent": "test-agent",
            "model": "anthropic:claude-haiku-4-5",
        }
        config_file = tmp_path / "deepagents.json"
        config_file.write_text(json.dumps(config_data))

        config = DeployConfig.load(config_file)
        assert config.agent == "test-agent"
        assert config.model == "anthropic:claude-haiku-4-5"

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        config = DeployConfig.load(tmp_path / "nonexistent.json")
        assert config.agent == "agent"
        assert config.model == "anthropic:claude-sonnet-4-6"

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "deepagents.json"
        config_file.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            DeployConfig.load(config_file)


# ------------------------------------------------------------------
# Namespace validation
# ------------------------------------------------------------------


class TestNamespaceConfig:
    def test_valid_scopes(self) -> None:
        for scope in ("assistant", "user", "thread", "user+thread"):
            ns = NamespaceConfig(scope=scope)
            assert ns.scope == scope

    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid namespace scope"):
            NamespaceConfig(scope="global")

    def test_from_dict_defaults(self) -> None:
        ns = NamespaceConfig.from_dict(None)
        assert ns.scope == "assistant"
        assert ns.prefix == "filesystem"


class TestBackendConfig:
    def test_valid_types(self) -> None:
        for backend_type in ("store", "sandbox", "custom"):
            if backend_type == "custom":
                bc = BackendConfig(type=backend_type, path="./my.py:factory")
            else:
                bc = BackendConfig(type=backend_type)
            assert bc.type == backend_type

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid backend type"):
            BackendConfig(type="redis")

    def test_custom_without_path_raises(self) -> None:
        with pytest.raises(ValueError, match="Custom backend requires"):
            BackendConfig(type="custom")


class TestSandboxConfig:
    def test_valid_providers(self) -> None:
        for provider in ("langsmith", "modal", "daytona", "runloop"):
            sc = SandboxConfig(provider=provider)
            assert sc.provider == provider

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid sandbox provider"):
            SandboxConfig(provider="e2b")

    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid sandbox scope"):
            SandboxConfig(scope="global")

    def test_from_dict_false_returns_none(self) -> None:
        assert SandboxConfig.from_dict(False) is None


class TestMemoryConfig:
    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid memory scope"):
            MemoryConfig(scope="global")


# ------------------------------------------------------------------
# Bundling
# ------------------------------------------------------------------


class TestBundleArtifacts:
    def test_basic_bundle(self, tmp_path: Path) -> None:
        """Test that bundling creates the expected files."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        # Create minimal project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        (deepagents_dir / "AGENTS.md").write_text("# My Agent\nBe helpful.")

        config = DeployConfig(
            memory=MemoryConfig(sources=[".deepagents/AGENTS.md"]),
            skills=SkillsConfig(sources=[]),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        # Check expected files
        assert (deploy_dir / "deploy_graph.py").exists()
        assert (deploy_dir / "deploy_config.json").exists()
        assert (deploy_dir / "langgraph.json").exists()
        assert (deploy_dir / "pyproject.toml").exists()
        assert (deploy_dir / "agents").is_dir()

        # Check langgraph.json
        lg_config = json.loads((deploy_dir / "langgraph.json").read_text())
        assert "agent" in lg_config["graphs"]
        assert lg_config["graphs"]["agent"] == "./deploy_graph.py:graph"
        assert lg_config["python_version"] == "3.12"

    def test_bundle_with_skills(self, tmp_path: Path) -> None:
        """Test bundling with skills directories."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create a skill
        skill_dir = project_root / ".deepagents" / "skills" / "code-review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: Review code\n---\n# Code Review\n"
        )

        config = DeployConfig(
            memory=MemoryConfig(sources=[]),
            skills=SkillsConfig(sources=[".deepagents/skills"]),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        assert (deploy_dir / "skills" / "code-review" / "SKILL.md").exists()

    def test_bundle_with_custom_tools(self, tmp_path: Path) -> None:
        """Test bundling with custom tools module."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create custom tools file
        (project_root / "my_tools.py").write_text(
            "from langchain_core.tools import tool\n\n"
            "@tool\ndef hello() -> str:\n    return 'hi'\n\n"
            "tools = [hello]\n"
        )

        config = DeployConfig(
            memory=MemoryConfig(sources=[]),
            skills=SkillsConfig(sources=[]),
            tools=ToolsConfig(custom="./my_tools.py:tools"),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        assert (deploy_dir / "my_tools.py").exists()
        dc = json.loads((deploy_dir / "deploy_config.json").read_text())
        assert dc["tools"]["_bundled_custom"] == "./my_tools.py:tools"

    def test_bundle_langgraph_json_store_index(self, tmp_path: Path) -> None:
        """Test that store backend generates store index in langgraph.json."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        project_root = tmp_path / "project"
        project_root.mkdir()

        config = DeployConfig(
            memory=MemoryConfig(sources=[]),
            skills=SkillsConfig(sources=[]),
            backend=BackendConfig(type="store"),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        lg_config = json.loads((deploy_dir / "langgraph.json").read_text())
        assert "store" in lg_config
        assert "index" in lg_config["store"]

    def test_bundle_sandbox_backend_no_store_index(self, tmp_path: Path) -> None:
        """Test that sandbox backend does NOT generate store index."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        project_root = tmp_path / "project"
        project_root.mkdir()

        config = DeployConfig(
            memory=MemoryConfig(sources=[]),
            skills=SkillsConfig(sources=[]),
            backend=BackendConfig(type="sandbox"),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        lg_config = json.loads((deploy_dir / "langgraph.json").read_text())
        assert "store" not in lg_config

    def test_bundle_deploy_config_scoping(self, tmp_path: Path) -> None:
        """Test that scoping config is properly serialized."""
        from deepagents_cli.deploy.bundle import bundle_deploy_artifacts

        project_root = tmp_path / "project"
        project_root.mkdir()

        config = DeployConfig(
            memory=MemoryConfig(scope="user", sources=[]),
            skills=SkillsConfig(sources=[]),
            backend=BackendConfig(
                namespace=NamespaceConfig(scope="user+thread", prefix="files"),
            ),
            sandbox=SandboxConfig(provider="langsmith", scope="assistant"),
        )

        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)

        dc = json.loads((deploy_dir / "deploy_config.json").read_text())
        assert dc["memory"]["scope"] == "user"
        assert dc["backend"]["namespace"]["scope"] == "user+thread"
        assert dc["backend"]["namespace"]["prefix"] == "files"
        assert dc["sandbox"]["scope"] == "assistant"
        assert dc["sandbox"]["provider"] == "langsmith"


# ------------------------------------------------------------------
# Namespace factory
# ------------------------------------------------------------------


def _import_deploy_graph_functions() -> tuple:
    """Import functions from deploy_graph without triggering module-level make_graph().

    The deploy_graph module calls make_graph() and sys.exit(1) at import time
    if deploy_config.json is missing. We use importlib to load just the functions.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    module_path = (
        Path(__file__).parent.parent.parent
        / "deepagents_cli"
        / "deploy"
        / "deploy_graph.py"
    )
    # Load as a different module name to avoid the cached import
    spec = importlib.util.spec_from_file_location(
        "deploy_graph_test_only", module_path, submodule_search_locations=[]
    )
    assert spec is not None
    assert spec.loader is not None

    # Read the source and extract only the functions we need
    source = module_path.read_text()

    # Create a minimal module with just the functions
    import types

    mod = types.ModuleType("deploy_graph_test_only")
    mod.__file__ = str(module_path)

    # Execute only the function definitions, not the module-level try/except
    # by stripping the last block
    lines = source.split("\n")
    # Find the line "try:" near the end
    cut_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "try:":
            cut_idx = i
            break
    safe_source = "\n".join(lines[:cut_idx])

    exec(compile(safe_source, str(module_path), "exec"), mod.__dict__)  # noqa: S102
    return mod._resolve_scope, mod._make_namespace_factory


class TestNamespaceFactory:
    def test_assistant_scope(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {"configurable": {"assistant_id": "my-agent"}}
        result = _resolve_scope("assistant", config)
        assert result == ("my-agent",)

    def test_user_scope(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {
            "configurable": {
                "langgraph_auth_user": {"identity": "user-123"},
            },
        }
        result = _resolve_scope("user", config)
        assert result == ("user-123",)

    def test_thread_scope(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {"configurable": {"thread_id": "thread-abc"}}
        result = _resolve_scope("thread", config)
        assert result == ("thread-abc",)

    def test_user_thread_scope(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {
            "configurable": {
                "langgraph_auth_user": {"identity": "user-123"},
                "thread_id": "thread-abc",
            },
        }
        result = _resolve_scope("user+thread", config)
        assert result == ("user-123", "thread-abc")

    def test_unknown_scope_falls_back(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {"configurable": {"assistant_id": "fallback"}}
        result = _resolve_scope("bogus", config)
        assert result == ("fallback",)

    def test_missing_auth_user_defaults(self) -> None:
        _resolve_scope, _ = _import_deploy_graph_functions()
        config = {"configurable": {}}
        result = _resolve_scope("user", config)
        assert result == ("default",)
