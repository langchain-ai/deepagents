"""Tests for the deploy module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli.deploy import (
    DeployConfig,
    _collect_agents_md,
    _collect_skills,
    _collect_user_agents_md,
    _write_deploy_graph,
    _write_env_file,
    _write_langgraph_json,
    _write_pyproject,
    scaffold_deployment,
)


class TestDeployConfig:
    def test_defaults(self):
        config = DeployConfig()
        assert config.sandbox_type == "langsmith"
        assert config.model is None
        assert config.agent_name == "agent"
        assert config.deployment_name is None
        assert config.dry_run is False

    def test_custom_values(self):
        config = DeployConfig(
            sandbox_type="modal",
            model="anthropic:claude-opus-4-6",
            agent_name="coder",
            deployment_name="my-deploy",
            dry_run=True,
        )
        assert config.sandbox_type == "modal"
        assert config.model == "anthropic:claude-opus-4-6"
        assert config.agent_name == "coder"
        assert config.deployment_name == "my-deploy"
        assert config.dry_run is True


class TestCollectAgentsMd:
    def test_no_project_root(self):
        assert _collect_agents_md(None) == []

    def test_with_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Project instructions")
        result = _collect_agents_md(tmp_path)
        assert len(result) == 1
        assert result[0] == ("AGENTS.md", "# Project instructions")

    def test_with_deepagents_agents_md(self, tmp_path):
        da_dir = tmp_path / ".deepagents"
        da_dir.mkdir()
        (da_dir / "AGENTS.md").write_text("# Deep agents config")
        result = _collect_agents_md(tmp_path)
        assert len(result) == 1
        assert result[0] == (".deepagents/AGENTS.md", "# Deep agents config")

    def test_both_locations(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Root")
        da_dir = tmp_path / ".deepagents"
        da_dir.mkdir()
        (da_dir / "AGENTS.md").write_text("# Nested")
        result = _collect_agents_md(tmp_path)
        assert len(result) == 2

    def test_no_files(self, tmp_path):
        assert _collect_agents_md(tmp_path) == []


class TestCollectSkills:
    def test_no_project_root(self):
        assert _collect_skills(None) == []

    def test_with_deepagents_skills(self, tmp_path):
        skills_dir = tmp_path / ".deepagents" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "my-skill").mkdir()
        result = _collect_skills(tmp_path)
        assert len(result) == 1
        assert result[0][0] == ".deepagents/skills"

    def test_with_agents_skills(self, tmp_path):
        skills_dir = tmp_path / ".agents" / "skills"
        skills_dir.mkdir(parents=True)
        result = _collect_skills(tmp_path)
        assert len(result) == 1
        assert result[0][0] == ".agents/skills"

    def test_no_skills(self, tmp_path):
        assert _collect_skills(tmp_path) == []


class TestCollectUserAgentsMd:
    def test_no_user_dir(self, tmp_path):
        with patch("deepagents_cli.deploy.Path.home", return_value=tmp_path):
            assert _collect_user_agents_md("agent") is None

    def test_with_user_agents_md(self, tmp_path):
        agent_dir = tmp_path / ".deepagents" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "AGENTS.md").write_text("# User config")
        with patch("deepagents_cli.deploy.Path.home", return_value=tmp_path):
            result = _collect_user_agents_md("agent")
        assert result == "# User config"

    def test_empty_user_agents_md(self, tmp_path):
        agent_dir = tmp_path / ".deepagents" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "AGENTS.md").write_text("")
        with patch("deepagents_cli.deploy.Path.home", return_value=tmp_path):
            assert _collect_user_agents_md("agent") is None


class TestWriteDeployGraph:
    def test_creates_file(self, tmp_path):
        path = _write_deploy_graph(tmp_path)
        assert path.exists()
        assert path.name == "deploy_graph.py"
        content = path.read_text()
        assert "async def get_agent" in content
        assert "create_deep_agent" in content
        assert "SANDBOX_TYPE" in content

    def test_contains_sandbox_providers(self, tmp_path):
        path = _write_deploy_graph(tmp_path)
        content = path.read_text()
        assert "langsmith" in content
        assert "daytona" in content
        assert "modal" in content
        assert "runloop" in content


class TestWriteLanggraphJson:
    def test_creates_valid_json(self, tmp_path):
        path = _write_langgraph_json(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "graphs" in data
        assert "agent" in data["graphs"]
        assert "deploy_graph.py:get_agent" in data["graphs"]["agent"]
        assert data["dependencies"] == ["."]

    def test_with_env_file(self, tmp_path):
        path = _write_langgraph_json(tmp_path, env_file="./.env")
        data = json.loads(path.read_text())
        assert data["env"] == "./.env"

    def test_python_version(self, tmp_path):
        path = _write_langgraph_json(tmp_path)
        data = json.loads(path.read_text())
        expected = f"3.{sys.version_info.minor}"
        assert data["python_version"] == expected


class TestWritePyproject:
    def test_creates_file(self, tmp_path):
        path = _write_pyproject(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "deepagents-deploy" in content
        assert "deepagents>=" in content
        assert "langsmith[sandbox]" in content

    def test_langsmith_sandbox(self, tmp_path):
        path = _write_pyproject(tmp_path, sandbox_type="langsmith")
        content = path.read_text()
        assert "langchain-daytona" not in content
        assert "langchain-modal" not in content

    def test_daytona_sandbox(self, tmp_path):
        path = _write_pyproject(tmp_path, sandbox_type="daytona")
        content = path.read_text()
        assert "langchain-daytona" in content

    def test_modal_sandbox(self, tmp_path):
        path = _write_pyproject(tmp_path, sandbox_type="modal")
        content = path.read_text()
        assert "langchain-modal" in content

    def test_runloop_sandbox(self, tmp_path):
        path = _write_pyproject(tmp_path, sandbox_type="runloop")
        content = path.read_text()
        assert "langchain-runloop" in content


class TestWriteEnvFile:
    def test_writes_sandbox_type(self, tmp_path):
        config = DeployConfig(sandbox_type="langsmith")
        path = _write_env_file(tmp_path, config=config)
        assert path is not None
        content = path.read_text()
        assert "SANDBOX_TYPE=langsmith" in content

    def test_writes_model(self, tmp_path):
        config = DeployConfig(model="anthropic:claude-opus-4-6")
        path = _write_env_file(tmp_path, config=config)
        assert path is not None
        content = path.read_text()
        assert "DEEPAGENTS_DEPLOY_MODEL=anthropic:claude-opus-4-6" in content

    def test_picks_up_env_vars(self, tmp_path):
        config = DeployConfig()
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "sk-test", "LANGSMITH_API_KEY": "ls-test"},
            clear=False,
        ):
            path = _write_env_file(tmp_path, config=config)
        assert path is not None
        content = path.read_text()
        assert "ANTHROPIC_API_KEY=sk-test" in content
        assert "LANGSMITH_API_KEY=ls-test" in content

    def test_merges_user_env_file(self, tmp_path):
        user_env = tmp_path / "user.env"
        user_env.write_text("CUSTOM_VAR=hello\n# comment\nANOTHER=world\n")
        config = DeployConfig()
        path = _write_env_file(tmp_path, config=config, user_env_file=str(user_env))
        assert path is not None
        content = path.read_text()
        assert "CUSTOM_VAR=hello" in content
        assert "ANOTHER=world" in content

    def test_api_key_from_config(self, tmp_path):
        config = DeployConfig(api_key="my-key")
        with patch.dict("os.environ", {}, clear=True):
            path = _write_env_file(tmp_path, config=config)
        assert path is not None
        content = path.read_text()
        assert "LANGSMITH_API_KEY=my-key" in content


class TestScaffoldDeployment:
    def test_creates_all_artifacts(self, tmp_path):
        config = DeployConfig()
        work_dir = scaffold_deployment(config, output_dir=tmp_path)
        assert (work_dir / "deploy_graph.py").exists()
        assert (work_dir / "langgraph.json").exists()
        assert (work_dir / "pyproject.toml").exists()
        assert (work_dir / "bundled").exists()

    def test_bundles_agents_md(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("# My project")

        output = tmp_path / "output"
        config = DeployConfig()
        work_dir = scaffold_deployment(config, project_root=project, output_dir=output)
        agents_md_dir = work_dir / "bundled" / "agents_md"
        assert agents_md_dir.exists()
        assert (agents_md_dir / "AGENTS.md").exists()
        assert (agents_md_dir / "AGENTS.md").read_text() == "# My project"

    def test_bundles_skills(self, tmp_path):
        project = tmp_path / "project"
        skills_dir = project / ".deepagents" / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# Skill")

        output = tmp_path / "output"
        config = DeployConfig()
        work_dir = scaffold_deployment(config, project_root=project, output_dir=output)
        bundled_skills = work_dir / "bundled" / "skills"
        assert bundled_skills.exists()

    def test_langgraph_json_valid(self, tmp_path):
        config = DeployConfig()
        work_dir = scaffold_deployment(config, output_dir=tmp_path)
        data = json.loads((work_dir / "langgraph.json").read_text())
        assert data["graphs"]["agent"] == "./deploy_graph.py:get_agent"

    def test_temp_dir_when_no_output(self):
        config = DeployConfig()
        work_dir = scaffold_deployment(config)
        try:
            assert work_dir.exists()
            assert (work_dir / "deploy_graph.py").exists()
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


class TestParseArgs:
    def test_deploy_subcommand(self):
        from deepagents_cli.main import parse_args

        with patch.object(sys, "argv", ["deepagents", "deploy"]):
            args = parse_args()
        assert args.command == "deploy"
        assert args.sandbox == "langsmith"
        assert args.dry_run is False

    def test_deploy_with_options(self):
        from deepagents_cli.main import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "deploy",
                "--sandbox",
                "modal",
                "--model",
                "gpt-5",
                "--name",
                "my-deploy",
                "--dry-run",
            ],
        ):
            args = parse_args()
        assert args.command == "deploy"
        assert args.sandbox == "modal"
        assert args.model == "gpt-5"
        assert args.deployment_name == "my-deploy"
        assert args.dry_run is True

    def test_dev_subcommand(self):
        from deepagents_cli.main import parse_args

        with patch.object(sys, "argv", ["deepagents", "dev"]):
            args = parse_args()
        assert args.command == "dev"
        assert args.sandbox == "langsmith"
        assert args.port == 2024
        assert args.host == "127.0.0.1"

    def test_dev_with_options(self):
        from deepagents_cli.main import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "dev",
                "--sandbox",
                "daytona",
                "--port",
                "8000",
                "--host",
                "0.0.0.0",
            ],
        ):
            args = parse_args()
        assert args.command == "dev"
        assert args.sandbox == "daytona"
        assert args.port == 8000
        assert args.host == "0.0.0.0"
