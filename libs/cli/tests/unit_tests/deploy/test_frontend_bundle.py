"""Tests for bundler behavior when [frontend].enabled = true."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.bundler import bundle
from deepagents_cli.deploy.config import (
    AgentConfig,
    AuthConfig,
    DeployConfig,
    FrontendConfig,
)


@pytest.fixture
def shipped_frontend_dist(tmp_path, monkeypatch):
    """Fake the shipped frontend_dist so tests don't require a real Vite build.

    Writes a minimal `index.html` with the placeholder and one asset file,
    then points the bundler's copy source at this directory.
    """
    fake_dist = tmp_path / "fake_frontend_dist"
    fake_dist.mkdir()
    assets = fake_dist / "assets"
    assets.mkdir()
    (assets / "index-abc.js").write_text("/* fake bundle */", encoding="utf-8")
    (fake_dist / "index.html").write_text(
        "<!doctype html>\n<html><head>"
        '<script>window.__DEEPAGENTS_CONFIG__ = {"__PLACEHOLDER__":true};</script>'
        '<script src="/app/assets/index-abc.js"></script>'
        '</head><body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    monkeypatch.setattr("deepagents_cli.deploy.bundler._FRONTEND_DIST_SRC", fake_dist)
    return fake_dist


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "AGENTS.md").write_text("prompt", encoding="utf-8")
    return proj


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    d = tmp_path / "build"
    d.mkdir()
    return d


def _supabase_config() -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="my-agent", model="anthropic:claude-sonnet-4-6"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True, app_name="My App"),
    )


def _clerk_config() -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="my-agent"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_emits_app_py(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon")
    bundle(_supabase_config(), project, build_dir)
    app_py = build_dir / "app.py"
    assert app_py.is_file()
    content = app_py.read_text(encoding="utf-8")
    assert "Starlette" in content
    assert "StaticFiles" in content


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_copies_frontend_dist(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon")
    bundle(_supabase_config(), project, build_dir)
    dest = build_dir / "frontend_dist"
    assert (dest / "index.html").is_file()
    assert (dest / "assets" / "index-abc.js").is_file()


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_rewrites_placeholder_supabase(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://xyz.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon-xyz")
    bundle(_supabase_config(), project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"supabase"' in html
    assert '"supabaseUrl":"https://xyz.supabase.co"' in html
    assert '"supabaseAnonKey":"anon-xyz"' in html
    assert '"appName":"My App"' in html


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_rewrites_placeholder_clerk(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_abc")
    bundle(_clerk_config(), project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"clerk"' in html
    assert '"clerkPublishableKey":"pk_test_abc"' in html


def test_bundle_without_frontend_still_works(
    project: Path,
    build_dir: Path,
) -> None:
    cfg = DeployConfig(agent=AgentConfig(name="my-agent"))
    bundle(cfg, project, build_dir)
    assert not (build_dir / "app.py").exists()
    assert not (build_dir / "frontend_dist").exists()


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_escapes_angle_bracket_in_app_name(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent `</script>` in app_name from breaking out of the inline script."""
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    cfg = DeployConfig(
        agent=AgentConfig(name="my-agent"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True, app_name="</script>hack"),
    )
    bundle(cfg, project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "\\u003c/script>" in html


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_bundle_succeeds_when_frontend_enabled_without_auth(
    project: Path,
    build_dir: Path,
) -> None:
    """[frontend].enabled without [auth] bundles in anonymous mode."""
    cfg = DeployConfig(
        agent=AgentConfig(name="my-agent"),
        frontend=FrontendConfig(enabled=True),
    )
    bundle(cfg, project, build_dir)

    # Anonymous auth.py is generated to override LangSmith Cloud's
    # default x-api-key requirement (otherwise the bundled frontend
    # can't reach /threads).
    auth_py = (build_dir / "auth.py").read_text(encoding="utf-8")
    assert "Anonymous-mode auth" in auth_py
    assert '"identity": "anonymous"' in auth_py

    # Frontend bundle copied with anonymous runtime config injected.
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert '"auth":"none"' in html
    assert "supabaseUrl" not in html
    assert "clerkPublishableKey" not in html


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_langgraph_json_has_http_app_when_frontend_enabled(
    project: Path,
    build_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    bundle(_supabase_config(), project, build_dir)
    data = json.loads((build_dir / "langgraph.json").read_text(encoding="utf-8"))
    assert data["http"] == {"app": "./app.py:app"}


def test_langgraph_json_no_http_app_when_frontend_disabled(
    project: Path,
    build_dir: Path,
) -> None:
    cfg = DeployConfig(agent=AgentConfig(name="my-agent"))
    bundle(cfg, project, build_dir)
    data = json.loads((build_dir / "langgraph.json").read_text(encoding="utf-8"))
    assert "http" not in data


def test_deploy_dry_run_supabase_end_to_end(tmp_path, monkeypatch, capsys):
    """End-to-end `_deploy(dry_run=True)` with the real shipped bundle."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "AGENTS.md").write_text("prompt", encoding="utf-8")
    (project / "deepagents.toml").write_text(
        """
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

[auth]
provider = "supabase"

[frontend]
enabled = true
app_name = "My App"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")

    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.chdir(project)
    _deploy(config_path=str(project / "deepagents.toml"), dry_run=True)

    out = capsys.readouterr().out
    assert "Inspect the build directory" in out
    build_line = [line for line in out.splitlines() if "build directory" in line][-1]
    build_path = Path(build_line.split("Inspect the build directory:")[-1].strip())
    assert (build_path / "app.py").is_file()
    assert (build_path / "frontend_dist" / "index.html").is_file()
    html = (build_path / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"supabase"' in html


def _write_anonymous_project(project: Path) -> None:
    """Helper: write an [agent]+[frontend] (no [auth]) project at *project*."""
    project.mkdir(exist_ok=True)
    (project / "AGENTS.md").write_text("prompt", encoding="utf-8")
    (project / "deepagents.toml").write_text(
        """
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

[frontend]
enabled = true
""",
        encoding="utf-8",
    )


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_deploy_dry_run_anonymous_prints_warning(tmp_path, monkeypatch, capsys):
    """`_deploy` with frontend enabled and no [auth] prints the new warning."""
    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_anonymous_project(tmp_path / "proj")

    _deploy(config_path=str(tmp_path / "proj" / "deepagents.toml"), dry_run=True)

    captured = capsys.readouterr()
    assert "ANONYMOUS auth" in captured.out
    assert "API is open to anyone" in captured.out
    # Summary line uses the new label.
    assert "Auth: anonymous (API open to anyone)" in captured.out


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_deploy_default_summary_label(tmp_path, monkeypatch, capsys):
    """No [auth] and no [frontend] prints the 'Auth: default' summary line."""
    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "AGENTS.md").write_text("prompt", encoding="utf-8")
    (project / "deepagents.toml").write_text(
        """
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"
""",
        encoding="utf-8",
    )

    _deploy(config_path=str(project / "deepagents.toml"), dry_run=True)

    captured = capsys.readouterr()
    assert "Auth: default (LangSmith API key required)" in captured.out
    # No anonymous warning fires for the default case.
    assert "ANONYMOUS auth" not in captured.out


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_deploy_anonymous_dry_run_skips_confirmation(tmp_path, monkeypatch):
    """`--dry-run` on an anonymous deploy never prompts."""
    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_anonymous_project(tmp_path / "proj")

    def _no_input(_prompt: str) -> str:
        msg = "input() should not be called during dry-run"
        raise AssertionError(msg)

    monkeypatch.setattr("builtins.input", _no_input)

    # Should not raise.
    _deploy(config_path=str(tmp_path / "proj" / "deepagents.toml"), dry_run=True)


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_deploy_anonymous_aborts_on_no(tmp_path, monkeypatch, capsys):
    """Answering anything other than y/yes aborts the deploy with exit 1."""
    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_anonymous_project(tmp_path / "proj")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(SystemExit) as excinfo:
        _deploy(config_path=str(tmp_path / "proj" / "deepagents.toml"), dry_run=False)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Aborted." in captured.out


@pytest.mark.usefixtures("shipped_frontend_dist")
def test_deploy_anonymous_aborts_on_eof(tmp_path, monkeypatch, capsys):
    """Closing stdin (EOFError) aborts cleanly with exit 1."""
    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_anonymous_project(tmp_path / "proj")

    def _raises_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raises_eof)

    with pytest.raises(SystemExit) as excinfo:
        _deploy(config_path=str(tmp_path / "proj" / "deepagents.toml"), dry_run=False)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Aborted." in captured.out


def test_build_runtime_config_json_anonymous_mode():
    """When config.auth is None, the runtime config has auth:'none'."""
    import json

    from deepagents_cli.deploy.bundler import _build_runtime_config_json

    cfg = DeployConfig(
        agent=AgentConfig(name="my-agent"),
        frontend=FrontendConfig(enabled=True, app_name="My App"),
    )
    payload_str = _build_runtime_config_json(cfg)
    payload = json.loads(payload_str)
    assert payload == {
        "auth": "none",
        "appName": "My App",
        "assistantId": "agent",
    }
