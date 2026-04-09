# Deploy Subagents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subagent support to `deepagents deploy` so nested subagent directories are auto-discovered, bundled, and wired into the deployed agent graph.

**Architecture:** Each subagent is a self-contained mini deploy unit under `agents/{name}/` with its own `AGENTS.md`, `deepagents.toml`, optional `skills/`, and optional `mcp.json`. The bundler discovers these, seeds their data into the store, and the generated `deploy_graph.py` passes them to `create_deep_agent(subagents=[...])`. The SDK gains a `memory` field on `SubAgent` so deployed subagents can load their own `AGENTS.md` as memory.

**Tech Stack:** Python 3.12, deepagents SDK, deepagents-cli, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-deploy-subagents-design.md`

---

### Task 1: Add `memory` field to SDK `SubAgent` TypedDict

**Files:**
- Modify: `libs/deepagents/deepagents/middleware/subagents.py:76-77`
- Modify: `libs/deepagents/deepagents/graph.py:337-342`

- [ ] **Step 1: Add `memory` field to `SubAgent` TypedDict**

In `libs/deepagents/deepagents/middleware/subagents.py`, add after line 77 (`skills` field):

```python
    memory: NotRequired[list[str]]
    """Memory source paths for MemoryMiddleware (e.g., ["/memories/agents/researcher/AGENTS.md"])."""
```

- [ ] **Step 2: Wire `MemoryMiddleware` into subagent middleware stack in `graph.py`**

In `libs/deepagents/deepagents/graph.py`, in the `SubAgent` processing branch (the `else` block starting at line 325), add memory middleware **after** user middleware and **before** the `AnthropicPromptCachingMiddleware` line (line 342).

Change lines 340-342 from:

```python
            subagent_middleware.extend(spec.get("middleware", []))
            # "ignore" skips caching for non-Anthropic models (see comment above).
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
```

To:

```python
            subagent_middleware.extend(spec.get("middleware", []))
            # "ignore" skips caching for non-Anthropic models (see comment above).
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
            subagent_memory = spec.get("memory")
            if subagent_memory:
                subagent_middleware.append(MemoryMiddleware(backend=backend, sources=subagent_memory))
```

This requires adding `MemoryMiddleware` to the imports at the top of `graph.py`. It's already imported — verify with the existing import on line ~27:

```python
from deepagents.middleware.memory import MemoryMiddleware
```

- [ ] **Step 3: Commit**

```bash
git add libs/deepagents/deepagents/middleware/subagents.py libs/deepagents/deepagents/graph.py
git commit -m "feat(sdk): add memory field to SubAgent TypedDict"
```

---

### Task 2: Add `SubagentConfig` and `load_subagents()` to deploy config

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Create: `libs/cli/tests/unit_tests/test_deploy_config.py`

- [ ] **Step 1: Write tests for subagent config loading**

Create `libs/cli/tests/unit_tests/test_deploy_config.py`:

```python
"""Unit tests for deploy subagent config loading."""

from pathlib import Path

import pytest

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    DeployConfig,
    SubagentConfig,
    load_subagents,
)


def _write_subagent(
    agents_dir: Path,
    name: str,
    *,
    description: str = "A test subagent",
    model: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str = "You are a test assistant.",
    include_toml: bool = True,
    include_skills: bool = False,
    include_mcp: bool = False,
    sandbox_provider: str | None = None,
) -> Path:
    """Helper to scaffold a subagent directory."""
    sa_dir = agents_dir / name
    sa_dir.mkdir(parents=True, exist_ok=True)

    # AGENTS.md with frontmatter
    (sa_dir / "AGENTS.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{system_prompt}\n"
    )

    if include_toml:
        sandbox_section = ""
        if sandbox_provider:
            sandbox_section = f"\n[sandbox]\nprovider = \"{sandbox_provider}\"\n"
        (sa_dir / "deepagents.toml").write_text(
            f'[agent]\nname = "{name}"\nmodel = "{model}"\n{sandbox_section}'
        )

    if include_skills:
        skill_dir = sa_dir / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\nDo the thing.\n"
        )

    if include_mcp:
        (sa_dir / "mcp.json").write_text(
            '{"mcpServers": {"test": {"type": "http", "url": "http://localhost:8080"}}}'
        )

    return sa_dir


class TestLoadSubagents:
    """Test load_subagents function."""

    def test_no_agents_dir(self, tmp_path: Path) -> None:
        """No agents/ directory returns empty list."""
        result = load_subagents(tmp_path)
        assert result == []

    def test_empty_agents_dir(self, tmp_path: Path) -> None:
        """Empty agents/ directory returns empty list."""
        (tmp_path / "agents").mkdir()
        result = load_subagents(tmp_path)
        assert result == []

    def test_single_subagent(self, tmp_path: Path) -> None:
        """Load a single subagent with toml."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "researcher")

        result = load_subagents(tmp_path)

        assert len(result) == 1
        sa = result[0]
        assert sa.agent.name == "researcher"
        assert sa.agent.model == "anthropic:claude-haiku-4-5-20251001"
        assert sa.description == "A test subagent"
        assert "test assistant" in sa.system_prompt

    def test_subagent_without_toml_uses_defaults(self, tmp_path: Path) -> None:
        """Subagent without deepagents.toml uses default model."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "researcher", include_toml=False)

        result = load_subagents(tmp_path)

        assert len(result) == 1
        sa = result[0]
        assert sa.agent.name == "researcher"
        # Default model from AgentConfig
        assert sa.agent.model == "anthropic:claude-sonnet-4-6"

    def test_subagent_sandbox_inheritance(self, tmp_path: Path) -> None:
        """Subagent without [sandbox] section has sandbox=None (inherit)."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "researcher")

        result = load_subagents(tmp_path)
        assert result[0].sandbox is None

    def test_subagent_sandbox_override(self, tmp_path: Path) -> None:
        """Subagent with [sandbox] section overrides parent."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "coder", sandbox_provider="modal")

        result = load_subagents(tmp_path)
        assert result[0].sandbox is not None
        assert result[0].sandbox.provider == "modal"

    def test_subagent_with_skills(self, tmp_path: Path) -> None:
        """Subagent skills/ directory is detected."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "researcher", include_skills=True)

        result = load_subagents(tmp_path)
        assert result[0].skills_dir is not None
        assert result[0].skills_dir.is_dir()

    def test_subagent_with_mcp(self, tmp_path: Path) -> None:
        """Subagent mcp.json is detected."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "researcher", include_mcp=True)

        result = load_subagents(tmp_path)
        assert result[0].mcp_path is not None
        assert result[0].mcp_path.is_file()

    def test_multiple_subagents_sorted(self, tmp_path: Path) -> None:
        """Multiple subagents are returned sorted by name."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "writer")
        _write_subagent(agents_dir, "researcher")

        result = load_subagents(tmp_path)
        assert len(result) == 2
        assert result[0].agent.name == "researcher"
        assert result[1].agent.name == "writer"

    def test_missing_agents_md_raises(self, tmp_path: Path) -> None:
        """Directory without AGENTS.md raises ValueError."""
        agents_dir = tmp_path / "agents"
        sa_dir = agents_dir / "broken"
        sa_dir.mkdir(parents=True)
        (sa_dir / "deepagents.toml").write_text('[agent]\nname = "broken"\n')

        with pytest.raises(ValueError, match="AGENTS.md not found"):
            load_subagents(tmp_path)

    def test_missing_frontmatter_description_raises(self, tmp_path: Path) -> None:
        """AGENTS.md without description raises ValueError."""
        agents_dir = tmp_path / "agents"
        sa_dir = agents_dir / "bad"
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text("---\nname: bad\n---\n\nNo description.\n")

        with pytest.raises(ValueError, match="description"):
            load_subagents(tmp_path)

    def test_name_mismatch_toml_vs_frontmatter_raises(self, tmp_path: Path) -> None:
        """Mismatched name between toml and AGENTS.md raises ValueError."""
        agents_dir = tmp_path / "agents"
        sa_dir = agents_dir / "researcher"
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text(
            "---\nname: researcher\ndescription: Research\n---\n\nPrompt.\n"
        )
        (sa_dir / "deepagents.toml").write_text('[agent]\nname = "wrong-name"\n')

        with pytest.raises(ValueError, match="mismatched name"):
            load_subagents(tmp_path)

    def test_reserved_name_general_purpose_raises(self, tmp_path: Path) -> None:
        """Subagent named 'general-purpose' raises ValueError."""
        agents_dir = tmp_path / "agents"
        _write_subagent(agents_dir, "general-purpose")

        with pytest.raises(ValueError, match="reserved"):
            load_subagents(tmp_path)

    def test_duplicate_names_raises(self, tmp_path: Path) -> None:
        """Two directories with same subagent name raises ValueError."""
        agents_dir = tmp_path / "agents"
        # Create two dirs that produce the same name
        _write_subagent(agents_dir, "researcher")
        # Create a second dir with a different folder name but same name in frontmatter
        sa_dir2 = agents_dir / "researcher-v2"
        sa_dir2.mkdir(parents=True)
        (sa_dir2 / "AGENTS.md").write_text(
            "---\nname: researcher\ndescription: Duplicate\n---\n\nPrompt.\n"
        )

        with pytest.raises(ValueError, match="Duplicate subagent name"):
            load_subagents(tmp_path)

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        """Files in agents/ are ignored (only directories are scanned)."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "stray-file.md").write_text("not a subagent")
        _write_subagent(agents_dir, "researcher")

        result = load_subagents(tmp_path)
        assert len(result) == 1

    def test_subagent_mcp_stdio_fails_validation(self, tmp_path: Path) -> None:
        """Subagent mcp.json with stdio transport fails deploy validation."""
        agents_dir = tmp_path / "agents"
        sa_dir = _write_subagent(agents_dir, "researcher", include_mcp=False)
        (sa_dir / "mcp.json").write_text(
            '{"mcpServers": {"local": {"type": "stdio", "command": "node"}}}'
        )

        # load_subagents succeeds (validation is separate)
        result = load_subagents(tmp_path)
        assert result[0].mcp_path is not None

        # But DeployConfig.validate() should catch it
        (tmp_path / "AGENTS.md").write_text("# Main agent\n")
        config = DeployConfig(
            agent=result[0].agent,
            subagents=result,
        )
        errors = config.validate(tmp_path)
        assert any("stdio" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_config.py -v
```

Expected: FAIL — `SubagentConfig` and `load_subagents` don't exist yet.

- [ ] **Step 3: Add `SubagentConfig` dataclass and `AGENTS_DIRNAME` constant to config.py**

In `libs/cli/deepagents_cli/deploy/config.py`, add after `MCP_FILENAME = "mcp.json"` (line 38):

```python
AGENTS_DIRNAME = "agents"
```

Add after `SandboxConfig` class (after line 72):

```python
@dataclass(frozen=True)
class SubagentConfig:
    """A single subagent parsed from agents/{name}/."""

    agent: AgentConfig
    sandbox: SandboxConfig | None  # None means inherit from parent
    system_prompt: str
    description: str
    skills_dir: Path | None  # absolute path to subagent's skills/ if present
    mcp_path: Path | None  # absolute path to subagent's mcp.json if present
```

- [ ] **Step 4: Add `subagents` field to `DeployConfig`**

Change the `DeployConfig` class (line 75-79) to add the `subagents` field:

```python
@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from `deepagents.toml`."""

    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    subagents: list[SubagentConfig] = field(default_factory=list)
```

- [ ] **Step 5: Add subagent validation to `DeployConfig.validate()`**

Add subagent validation at the end of the `validate` method (before `return errors`), around line 133:

```python
        # Validate subagents.
        seen_names: set[str] = set()
        for sa in self.subagents:
            if sa.agent.name in seen_names:
                errors.append(f"Duplicate subagent name: '{sa.agent.name}'")
            seen_names.add(sa.agent.name)

            if sa.agent.name == "general-purpose":
                errors.append("Subagent name 'general-purpose' is reserved")

            # Validate subagent MCP config (http/sse only).
            if sa.mcp_path and sa.mcp_path.is_file():
                errors.extend(_validate_mcp_for_deploy(sa.mcp_path))

            # Validate model credentials for subagent.
            errors.extend(_validate_model_credentials(sa.agent.model))

            # Validate subagent sandbox if overridden.
            if sa.sandbox is not None:
                if sa.sandbox.provider not in VALID_SANDBOX_PROVIDERS:
                    errors.append(
                        f"Subagent '{sa.agent.name}': unknown sandbox provider "
                        f"{sa.sandbox.provider}. Valid: {', '.join(sorted(VALID_SANDBOX_PROVIDERS))}"
                    )
```

- [ ] **Step 6: Implement `load_subagents()` function**

Add at the end of `config.py` (before the starter generators, around line 278):

```python
import re

import yaml


def _parse_subagent_frontmatter(agents_md_path: Path) -> tuple[str, str, str]:
    """Parse AGENTS.md frontmatter for name, description, and system_prompt.

    Returns:
        Tuple of (name, description, system_prompt).

    Raises:
        ValueError: If frontmatter is missing or invalid.
    """
    content = agents_md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        msg = f"{agents_md_path}: missing YAML frontmatter (--- delimiters)"
        raise ValueError(msg)

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        msg = f"{agents_md_path}: invalid YAML frontmatter: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(frontmatter, dict):
        msg = f"{agents_md_path}: frontmatter must be a YAML mapping"
        raise ValueError(msg)

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        msg = f"{agents_md_path}: missing required frontmatter field 'name'"
        raise ValueError(msg)

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description:
        msg = f"{agents_md_path}: missing required frontmatter field 'description'"
        raise ValueError(msg)

    system_prompt = match.group(2).strip()
    return name, description, system_prompt


def _load_subagent_toml(toml_path: Path) -> dict[str, Any]:
    """Load a subagent's deepagents.toml, returning raw dict.

    Returns empty dict if file doesn't exist.
    """
    if not toml_path.exists():
        return {}
    with toml_path.open("rb") as f:
        return tomllib.load(f)


def load_subagents(project_root: Path) -> list[SubagentConfig]:
    """Discover and load subagent configs from agents/ directory.

    Args:
        project_root: Directory containing the main deepagents.toml.

    Returns:
        List of SubagentConfig, sorted by name.

    Raises:
        ValueError: If a subagent has invalid config.
    """
    agents_dir = project_root / AGENTS_DIRNAME
    if not agents_dir.is_dir():
        return []

    subagents: list[SubagentConfig] = []
    seen_names: set[str] = set()

    for entry in sorted(agents_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Require AGENTS.md
        agents_md_path = entry / AGENTS_MD_FILENAME
        if not agents_md_path.is_file():
            msg = f"agents/{entry.name}/AGENTS.md not found — every subagent directory must contain an AGENTS.md"
            raise ValueError(msg)

        # Parse frontmatter
        fm_name, description, system_prompt = _parse_subagent_frontmatter(agents_md_path)

        # Load optional toml
        toml_data = _load_subagent_toml(entry / DEFAULT_CONFIG_FILENAME)

        # Build AgentConfig from toml (with frontmatter name as fallback)
        agent_data = toml_data.get("agent", {})
        toml_name = agent_data.get("name", fm_name)

        # Validate name consistency
        if toml_data and "agent" in toml_data and toml_name != fm_name:
            msg = (
                f"Subagent '{fm_name}' in agents/{entry.name}/ has mismatched "
                f"name in deepagents.toml: '{toml_name}'"
            )
            raise ValueError(msg)

        # Check reserved name
        if fm_name == "general-purpose":
            msg = "Subagent name 'general-purpose' is reserved"
            raise ValueError(msg)

        # Check for duplicates
        if fm_name in seen_names:
            msg = f"Duplicate subagent name: '{fm_name}'"
            raise ValueError(msg)
        seen_names.add(fm_name)

        agent_config = AgentConfig(
            name=fm_name,
            model=agent_data.get("model", "anthropic:claude-sonnet-4-6"),
        )

        # Sandbox: None means inherit from parent
        sandbox_data = toml_data.get("sandbox")
        sandbox_config: SandboxConfig | None = None
        if sandbox_data is not None:
            sandbox_config = SandboxConfig(
                provider=sandbox_data.get("provider", "none"),
                template=sandbox_data.get("template", "deepagents-deploy"),
                image=sandbox_data.get("image", "python:3"),
                scope=sandbox_data.get("scope", "thread"),
            )

        # Detect optional dirs/files
        skills_dir = entry / SKILLS_DIRNAME
        mcp_path = entry / MCP_FILENAME

        subagents.append(
            SubagentConfig(
                agent=agent_config,
                sandbox=sandbox_config,
                system_prompt=system_prompt,
                description=description,
                skills_dir=skills_dir if skills_dir.is_dir() else None,
                mcp_path=mcp_path if mcp_path.is_file() else None,
            )
        )

    return subagents
```

Also add `import re` and `import yaml` to the top of the file (yaml is already a dependency of deepagents-cli via PyYAML).

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_config.py -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py libs/cli/tests/unit_tests/test_deploy_config.py
git commit -m "feat(cli): add SubagentConfig and load_subagents to deploy config"
```

---

### Task 3: Extend bundler to include subagent data

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Create: `libs/cli/tests/unit_tests/test_deploy_bundler.py`

- [ ] **Step 1: Write tests for subagent bundling**

Create `libs/cli/tests/unit_tests/test_deploy_bundler.py`:

```python
"""Unit tests for deploy bundler with subagent support."""

import json
from pathlib import Path

from deepagents_cli.deploy.bundler import _build_seed, bundle
from deepagents_cli.deploy.config import (
    AgentConfig,
    DeployConfig,
    SandboxConfig,
    SubagentConfig,
)


def _make_project(tmp_path: Path, *, subagents: list[dict] | None = None) -> Path:
    """Scaffold a minimal deploy project."""
    (tmp_path / "AGENTS.md").write_text("# Main Agent\nYou are the main agent.\n")
    (tmp_path / "deepagents.toml").write_text(
        '[agent]\nname = "test-agent"\nmodel = "anthropic:claude-sonnet-4-6"\n'
    )

    for sa in subagents or []:
        sa_dir = tmp_path / "agents" / sa["name"]
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text(
            f"---\nname: {sa['name']}\ndescription: {sa.get('description', 'Test')}\n---\n\n"
            f"{sa.get('system_prompt', 'You are a test subagent.')}\n"
        )
        if sa.get("skills"):
            skill_dir = sa_dir / "skills" / "s1"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: s1\ndescription: Skill\n---\n\nDo it.\n")
        if sa.get("mcp"):
            (sa_dir / "mcp.json").write_text(
                '{"mcpServers": {"test": {"type": "http", "url": "http://localhost:8080"}}}'
            )

    return tmp_path


class TestBuildSeedWithSubagents:
    """Test _build_seed includes subagent data."""

    def test_seed_without_subagents(self, tmp_path: Path) -> None:
        """Seed with no subagents has empty subagents dict."""
        project = _make_project(tmp_path)
        config = DeployConfig(agent=AgentConfig(name="test-agent"))
        seed = _build_seed(config, project, "# Main Agent\n")

        assert "subagents" in seed
        assert seed["subagents"] == {}

    def test_seed_with_subagent(self, tmp_path: Path) -> None:
        """Seed includes subagent system_prompt, description, and memories."""
        project = _make_project(
            tmp_path,
            subagents=[{"name": "researcher", "description": "Research stuff"}],
        )
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher", model="anthropic:claude-haiku-4-5-20251001"),
            sandbox=None,
            system_prompt="You are a test subagent.",
            description="Research stuff",
            skills_dir=None,
            mcp_path=None,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )
        seed = _build_seed(config, project, "# Main Agent\n")

        assert "researcher" in seed["subagents"]
        sa_seed = seed["subagents"]["researcher"]
        assert sa_seed["system_prompt"] == "You are a test subagent."
        assert sa_seed["description"] == "Research stuff"
        assert "/AGENTS.md" in sa_seed["memories"]

    def test_seed_with_subagent_skills(self, tmp_path: Path) -> None:
        """Seed includes subagent skills."""
        project = _make_project(
            tmp_path,
            subagents=[{"name": "researcher", "skills": True}],
        )
        skills_dir = project / "agents" / "researcher" / "skills"
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"),
            sandbox=None,
            system_prompt="Test.",
            description="Test",
            skills_dir=skills_dir,
            mcp_path=None,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )
        seed = _build_seed(config, project, "# Main Agent\n")

        sa_seed = seed["subagents"]["researcher"]
        assert len(sa_seed["skills"]) > 0
        assert any("SKILL.md" in k for k in sa_seed["skills"])


class TestBundleWithSubagents:
    """Test full bundle() with subagents."""

    def test_bundle_copies_subagent_mcp(self, tmp_path: Path) -> None:
        """Subagent mcp.json is copied as _mcp_{name}.json."""
        project = _make_project(
            tmp_path,
            subagents=[{"name": "researcher", "mcp": True}],
        )
        mcp_path = project / "agents" / "researcher" / "mcp.json"
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"),
            sandbox=None,
            system_prompt="Test.",
            description="Test",
            skills_dir=None,
            mcp_path=mcp_path,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )

        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)

        assert (build_dir / "_mcp_researcher.json").is_file()

    def test_bundle_seed_has_subagents_key(self, tmp_path: Path) -> None:
        """Bundled _seed.json contains subagents."""
        project = _make_project(
            tmp_path,
            subagents=[{"name": "researcher"}],
        )
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"),
            sandbox=None,
            system_prompt="Test.",
            description="Test",
            skills_dir=None,
            mcp_path=None,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )

        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)

        seed = json.loads((build_dir / "_seed.json").read_text())
        assert "subagents" in seed
        assert "researcher" in seed["subagents"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_bundler.py -v
```

Expected: FAIL — `_build_seed` doesn't accept/return subagent data yet.

- [ ] **Step 3: Extend `_build_seed()` to include subagent data**

In `libs/cli/deepagents_cli/deploy/bundler.py`, modify `_build_seed` (line 110-147) to accept subagent configs and return subagent data:

```python
def _build_seed(
    config: DeployConfig,
    project_root: Path,
    system_prompt: str,
) -> dict[str, dict[str, str] | dict[str, dict]]:
    """Build the `_seed.json` payload.

    Layout:

    ```txt
    {
        "memories": { "AGENTS.md": "..." },
        "skills":   { "<skill>/SKILL.md": "...", ... },
        "subagents": {
            "<name>": {
                "system_prompt": "...",
                "description": "...",
                "memories": { "/AGENTS.md": "..." },
                "skills": { "/<skill>/SKILL.md": "...", ... }
            }
        }
    }
    ```
    """
    memories: dict[str, str] = {f"/{AGENTS_MD_FILENAME}": system_prompt}
    skills: dict[str, str] = {}

    skills_dir = project_root / SKILLS_DIRNAME
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skills_dir).as_posix()
                skills[f"/{rel}"] = f.read_text(encoding="utf-8")

    # Build subagent seed data.
    subagents_seed: dict[str, dict] = {}
    for sa in config.subagents:
        sa_memories: dict[str, str] = {
            f"/{AGENTS_MD_FILENAME}": sa.system_prompt,
        }
        sa_skills: dict[str, str] = {}
        if sa.skills_dir and sa.skills_dir.is_dir():
            for f in sorted(sa.skills_dir.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(sa.skills_dir).as_posix()
                    sa_skills[f"/{rel}"] = f.read_text(encoding="utf-8")

        subagents_seed[sa.agent.name] = {
            "system_prompt": sa.system_prompt,
            "description": sa.description,
            "memories": sa_memories,
            "skills": sa_skills,
        }

    return {"memories": memories, "skills": skills, "subagents": subagents_seed}
```

- [ ] **Step 4: Extend `bundle()` to copy subagent MCP files**

In the `bundle()` function (line 49-107), add after the main MCP copy step (after line 76) and before the deploy_graph render:

```python
    # 3c. Copy subagent mcp.json files.
    for sa in config.subagents:
        if sa.mcp_path and sa.mcp_path.is_file():
            dest = build_dir / f"_mcp_{sa.agent.name}.json"
            shutil.copy2(sa.mcp_path, dest)
            logger.info("Copied subagent %s mcp.json → %s", sa.agent.name, dest.name)
```

- [ ] **Step 5: Update seed logging to include subagent counts**

Update the log line after writing `_seed.json` (line 68-71):

```python
    logger.info(
        "Wrote _seed.json (memories: %d, skills: %d, subagents: %d)",
        len(seed["memories"]),
        len(seed["skills"]),
        len(seed["subagents"]),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_bundler.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/test_deploy_bundler.py
git commit -m "feat(cli): extend bundler to include subagent data in seed and copy mcp files"
```

---

### Task 4: Extend `DEPLOY_GRAPH_TEMPLATE` for subagents

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/templates.py`
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py` (update `_render_deploy_graph` call)

This is the most complex task. The generated `deploy_graph.py` must construct subagent specs at runtime and pass them to `create_deep_agent(subagents=[...])`.

- [ ] **Step 1: Add subagent config data structure to the template**

In `libs/cli/deepagents_cli/deploy/templates.py`, add a new template constant for the subagent section that gets inserted into `DEPLOY_GRAPH_TEMPLATE`. The approach: the bundler will render a Python literal list of subagent config dicts directly into the template.

Add before `DEPLOY_GRAPH_TEMPLATE` (around line 213):

```python
SUBAGENT_MCP_LOADER_TEMPLATE = '''\
async def _load_mcp_tools_{name}():
    """Load MCP tools for subagent '{name}'."""
    import json
    from pathlib import Path

    mcp_path = Path(__file__).parent / "_mcp_{name}.json"
    if not mcp_path.exists():
        return []

    try:
        raw = json.loads(mcp_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse _mcp_{name}.json: %s", exc)
        return []

    servers = raw.get("mcpServers", {{}})
    connections = {{}}
    for sname, cfg in servers.items():
        transport = cfg.get("type", cfg.get("transport", "stdio"))
        if transport in ("http", "sse"):
            conn = {{"transport": transport, "url": cfg["url"]}}
            if "headers" in cfg:
                conn["headers"] = cfg["headers"]
            connections[sname] = conn

    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(connections)
        return await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load MCP tools for subagent '{name}' from %d server(s): %s",
            len(connections),
            exc,
        )
        return []
'''
```

- [ ] **Step 2: Update `DEPLOY_GRAPH_TEMPLATE` to accept subagent data**

Modify the `make_graph()` function in `DEPLOY_GRAPH_TEMPLATE` to include subagent construction. Add a `{subagent_mcp_loaders}` placeholder before the `make_graph` function, and modify `make_graph` to build and pass subagents.

The key changes to the template (showing the end of the template around the `make_graph` function):

Replace the `make_graph` function and everything after `{mcp_tools_block}` with:

```python
{mcp_tools_block}

{subagent_mcp_loaders}

SUBAGENT_CONFIGS = {subagent_configs_literal}

...existing code up to make_graph...

async def make_graph(config: RunnableConfig, runtime: "ServerRuntime"):
    ...existing docstring and setup...

    # Seed subagent namespaces too.
    # (The seeding function is extended to handle subagent data.)

    tools: list = []
    {mcp_tools_load_call}

    backend_factory = _build_backend_factory(assistant_id)

    # Build subagent specs from bundled config.
    subagents = []
    for sa_cfg in SUBAGENT_CONFIGS:
        sa_tools = list(tools)  # inherit main agent tools
        mcp_loader_name = f"_load_mcp_tools_{{sa_cfg['name']}}"
        mcp_loader = globals().get(mcp_loader_name)
        if mcp_loader:
            sa_tools.extend(await mcp_loader())

        sa_name = sa_cfg["name"]
        sa_memory_prefix = f"{{MEMORIES_PREFIX}}agents/{{sa_name}}/"
        sa_skills_prefix = f"{{SKILLS_PREFIX}}agents/{{sa_name}}/"

        subagents.append({{
            "name": sa_name,
            "description": sa_cfg["description"],
            "system_prompt": sa_cfg["system_prompt"],
            "model": sa_cfg.get("model", {model!r}),
            "tools": sa_tools,
            "memory": [f"{{sa_memory_prefix}}AGENTS.md"],
            "skills": [sa_skills_prefix],
        }})

    return create_deep_agent(
        model={model!r},
        system_prompt=SYSTEM_PROMPT,
        memory=[f"{{MEMORIES_PREFIX}}AGENTS.md"],
        skills=[SKILLS_PREFIX],
        tools=tools,
        backend=backend_factory,
        subagents=subagents,
        middleware=[
            SandboxSyncMiddleware(backend=backend_factory, sources=[SKILLS_PREFIX]),
        ],
    )
```

And update `_seed_store_if_needed` to also seed subagent namespaces:

```python
async def _seed_store_if_needed(store, assistant_id: str) -> None:
    """Seed memories + skills + subagents under ``assistant_id`` once per process."""
    if assistant_id in _SEEDED_ASSISTANTS:
        return
    _SEEDED_ASSISTANTS.add(assistant_id)

    seed = _load_seed()

    memories_ns = (assistant_id, "memories")
    for path, content in seed.get("memories", {{}}).items():
        if await store.aget(memories_ns, path) is None:
            await store.aput(memories_ns, path, {{"content": content, "encoding": "utf-8"}})

    skills_ns = (assistant_id, "skills")
    for path, content in seed.get("skills", {{}}).items():
        if await store.aget(skills_ns, path) is None:
            await store.aput(skills_ns, path, {{"content": content, "encoding": "utf-8"}})

    # Seed subagent namespaces.
    for sa_name, sa_data in seed.get("subagents", {{}}).items():
        sa_mem_ns = (assistant_id, "memories", "agents", sa_name)
        for path, content in sa_data.get("memories", {{}}).items():
            if await store.aget(sa_mem_ns, path) is None:
                await store.aput(sa_mem_ns, path, {{"content": content, "encoding": "utf-8"}})

        sa_skills_ns = (assistant_id, "skills", "agents", sa_name)
        for path, content in sa_data.get("skills", {{}}).items():
            if await store.aget(sa_skills_ns, path) is None:
                await store.aput(sa_skills_ns, path, {{"content": content, "encoding": "utf-8"}})
```

Since this template is complex, the actual implementation should modify `DEPLOY_GRAPH_TEMPLATE` in place. The full updated template is large — the engineer should:

1. Add `{subagent_mcp_loaders}` placeholder after `{mcp_tools_block}`
2. Add `SUBAGENT_CONFIGS = {subagent_configs_literal}` as a module-level constant
3. Extend `_seed_store_if_needed` with the subagent namespace seeding loop
4. Extend `make_graph` to build subagent specs and pass `subagents=subagents` to `create_deep_agent`

- [ ] **Step 3: Update `_render_deploy_graph()` in bundler.py**

In `libs/cli/deepagents_cli/deploy/bundler.py`, modify `_render_deploy_graph` to accept subagent configs and render the new template placeholders:

```python
def _render_deploy_graph(
    config: DeployConfig,
    system_prompt: str,
    *,
    mcp_present: bool,
) -> str:
    """Render the generated `deploy_graph.py`."""
    provider = config.sandbox.provider
    if provider not in SANDBOX_BLOCKS:
        msg = f"Unknown sandbox provider {provider!r}. Valid: {sorted(SANDBOX_BLOCKS)}"
        raise ValueError(msg)
    sandbox_block, _ = SANDBOX_BLOCKS[provider]

    if mcp_present:
        mcp_tools_block = MCP_TOOLS_TEMPLATE
        mcp_tools_load_call = "tools.extend(await _load_mcp_tools())"
    else:
        mcp_tools_block = ""
        mcp_tools_load_call = "pass  # no MCP servers configured"

    # Render per-subagent MCP loaders.
    subagent_mcp_loaders = ""
    for sa in config.subagents:
        if sa.mcp_path and sa.mcp_path.is_file():
            subagent_mcp_loaders += SUBAGENT_MCP_LOADER_TEMPLATE.format(name=sa.agent.name)

    # Build subagent configs literal for embedding in generated code.
    subagent_configs = []
    for sa in config.subagents:
        sa_model = sa.agent.model
        subagent_configs.append({
            "name": sa.agent.name,
            "description": sa.description,
            "system_prompt": sa.system_prompt,
            "model": sa_model,
        })
    subagent_configs_literal = repr(subagent_configs)

    return DEPLOY_GRAPH_TEMPLATE.format(
        model=config.agent.model,
        system_prompt=system_prompt,
        sandbox_template=config.sandbox.template,
        sandbox_image=config.sandbox.image,
        sandbox_scope=config.sandbox.scope,
        sandbox_block=sandbox_block,
        mcp_tools_block=mcp_tools_block,
        mcp_tools_load_call=mcp_tools_load_call,
        default_assistant_id=config.agent.name,
        subagent_mcp_loaders=subagent_mcp_loaders,
        subagent_configs_literal=subagent_configs_literal,
    )
```

Import `SUBAGENT_MCP_LOADER_TEMPLATE` from templates at the top of bundler.py.

- [ ] **Step 4: Update `_render_pyproject()` to infer subagent deps**

In `_render_pyproject`, extend dep inference to include subagent model providers and MCP:

```python
def _render_pyproject(config: DeployConfig, *, mcp_present: bool) -> str:
    """Render the deployment package's `pyproject.toml`."""
    deps: list[str] = []

    # Collect all model strings (main + subagents).
    all_models = [config.agent.model] + [sa.agent.model for sa in config.subagents]
    for model_str in all_models:
        provider_prefix = model_str.split(":", 1)[0] if ":" in model_str else ""
        if provider_prefix and provider_prefix in _MODEL_PROVIDER_DEPS:
            dep = _MODEL_PROVIDER_DEPS[provider_prefix]
            if dep not in deps:
                deps.append(dep)

    # MCP: main agent or any subagent with mcp.
    any_mcp = mcp_present or any(
        sa.mcp_path and sa.mcp_path.is_file() for sa in config.subagents
    )
    if any_mcp:
        deps.append("langchain-mcp-adapters")

    _, partner_pkg = SANDBOX_BLOCKS.get(config.sandbox.provider, (None, None))
    if partner_pkg:
        deps.append(partner_pkg)

    extra_deps_lines = "".join(f'    "{dep}",\n' for dep in deps)

    return PYPROJECT_TEMPLATE.format(
        agent_name=config.agent.name,
        extra_deps=extra_deps_lines,
    )
```

- [ ] **Step 5: Verify the template renders valid Python**

Write a quick smoke test — add to `test_deploy_bundler.py`:

```python
class TestDeployGraphRendering:
    """Test that the rendered deploy_graph.py is valid Python."""

    def test_render_with_subagents_is_valid_python(self, tmp_path: Path) -> None:
        """Generated deploy_graph.py with subagents parses as valid Python."""
        import ast

        from deepagents_cli.deploy.bundler import _render_deploy_graph

        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher", model="anthropic:claude-haiku-4-5-20251001"),
            sandbox=None,
            system_prompt="You are a researcher.",
            description="Research stuff",
            skills_dir=None,
            mcp_path=None,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )
        result = _render_deploy_graph(config, "Main prompt", mcp_present=False)

        # Should parse without syntax errors.
        ast.parse(result)

        # Should contain subagent config.
        assert "researcher" in result
        assert "SUBAGENT_CONFIGS" in result

    def test_render_with_subagent_mcp_is_valid_python(self, tmp_path: Path) -> None:
        """Generated deploy_graph.py with subagent MCP parses as valid Python."""
        import ast

        from deepagents_cli.deploy.bundler import _render_deploy_graph

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text('{"mcpServers": {"test": {"type": "http", "url": "http://localhost"}}}')

        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher", model="openai:gpt-4o"),
            sandbox=None,
            system_prompt="You are a researcher.",
            description="Research stuff",
            skills_dir=None,
            mcp_path=mcp_file,
        )
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            subagents=[sa_config],
        )
        result = _render_deploy_graph(config, "Main prompt", mcp_present=False)

        ast.parse(result)
        assert "_load_mcp_tools_researcher" in result
```

- [ ] **Step 6: Run all bundler tests**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_bundler.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/templates.py libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/test_deploy_bundler.py
git commit -m "feat(cli): extend deploy_graph template and bundler for subagents"
```

---

### Task 5: Wire subagent loading into deploy commands

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/commands.py`

- [ ] **Step 1: Update `_deploy()` to load subagents**

In `libs/cli/deepagents_cli/deploy/commands.py`, modify the `_deploy()` function. After loading the config (around line 240-250), add subagent loading:

```python
    from deepagents_cli.deploy.config import load_subagents

    # ...after config = load_config(config_path)...
    config = DeployConfig(
        agent=config.agent,
        sandbox=config.sandbox,
        subagents=load_subagents(project_root),
    )
```

Where `project_root` is `config_path.parent` (the directory containing `deepagents.toml`).

- [ ] **Step 2: Update `_dev()` to load subagents**

Apply the same change to `_dev()` (the local dev server command) so it's consistent.

- [ ] **Step 3: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/commands.py
git commit -m "feat(cli): wire subagent loading into deploy and dev commands"
```

---

### Task 6: Update `deepagents init` to scaffold example subagent

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py` (add starter generators)
- Modify: `libs/cli/deepagents_cli/deploy/commands.py` (update `_init()`)

- [ ] **Step 1: Add starter generators for subagent files**

In `libs/cli/deepagents_cli/deploy/config.py`, add after the existing starter generators (after `generate_starter_mcp_json`, around line 335):

```python
STARTER_SUBAGENT_NAME = "researcher"


def generate_starter_subagent_agents_md() -> str:
    """Generate a starter subagent `AGENTS.md`."""
    return """\
---
name: researcher
description: Research topics on the web before writing content
---

You are a research assistant. Search for relevant information
and summarize your findings clearly and concisely.
"""


def generate_starter_subagent_config() -> str:
    """Generate a starter subagent `deepagents.toml`."""
    return """\
[agent]
name = "researcher"
model = "anthropic:claude-haiku-4-5-20251001"
"""
```

- [ ] **Step 2: Update `_init()` to create subagent directory**

In `libs/cli/deepagents_cli/deploy/commands.py`, modify the `_init()` function to scaffold the example subagent. After the skills directory creation (around line 207), add:

```python
    # Create agents/ directory with a starter subagent.
    from deepagents_cli.deploy.config import (
        AGENTS_DIRNAME,
        STARTER_SUBAGENT_NAME,
        generate_starter_subagent_agents_md,
        generate_starter_subagent_config,
    )

    agents_dir = project_dir / AGENTS_DIRNAME
    starter_subagent_dir = agents_dir / STARTER_SUBAGENT_NAME
    starter_subagent_dir.mkdir(parents=True, exist_ok=True)
    (starter_subagent_dir / AGENTS_MD_FILENAME).write_text(
        generate_starter_subagent_agents_md()
    )
    (starter_subagent_dir / DEFAULT_CONFIG_FILENAME).write_text(
        generate_starter_subagent_config()
    )
```

Also update the "Created ..." output to show the new files:

```python
    print(f"  {AGENTS_DIRNAME}/")
    print(f"    {STARTER_SUBAGENT_NAME}/AGENTS.md")
    print(f"    {STARTER_SUBAGENT_NAME}/deepagents.toml")
```

- [ ] **Step 3: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py libs/cli/deepagents_cli/deploy/commands.py
git commit -m "feat(cli): scaffold example subagent in deepagents init"
```

---

### Task 7: Update `print_bundle_summary()` for subagents

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`

- [ ] **Step 1: Extend `print_bundle_summary()` to show subagents**

In `libs/cli/deepagents_cli/deploy/bundler.py`, modify `print_bundle_summary` (line 227-259) to display subagent info after the model line:

```python
def print_bundle_summary(config: DeployConfig, build_dir: Path) -> None:
    """Print a human-readable summary of what was bundled."""
    seed_path = build_dir / "_seed.json"
    seed: dict = {"memories": {}, "skills": {}, "subagents": {}}
    if seed_path.exists():
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print(f"\n  Agent: {config.agent.name}")
    print(f"  Model: {config.agent.model}")

    # Subagents summary.
    if config.subagents:
        print(f"\n  Subagents ({len(config.subagents)}):")
        for sa in config.subagents:
            sa_seed = seed.get("subagents", {}).get(sa.agent.name, {})
            skills_count = len(sa_seed.get("skills", {}))
            has_mcp = (build_dir / f"_mcp_{sa.agent.name}.json").exists()
            print(f"    {sa.agent.name} ({sa.agent.model})")
            print(f"      skills: {skills_count}, mcp: {'yes' if has_mcp else 'no'}")

    memory_files = sorted(seed.get("memories", {}).keys())
    if memory_files:
        print(f"\n  Memory seed ({len(memory_files)} file(s)):")
        for f in memory_files:
            print(f"    {f}")

    skills_files = sorted(seed.get("skills", {}).keys())
    if skills_files:
        print(f"\n  Skills seed ({len(skills_files)} file(s)):")
        for f in skills_files:
            print(f"    {f}")

    if (build_dir / "_mcp.json").exists():
        print("\n  MCP config: _mcp.json")

    print(f"\n  Sandbox: {config.sandbox.provider}")
    print(f"\n  Build directory: {build_dir}")
    generated = sorted(f.name for f in build_dir.iterdir() if f.is_file())
    print(f"  Generated files: {', '.join(generated)}")
    print()
```

- [ ] **Step 2: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py
git commit -m "feat(cli): show subagents in deploy bundle summary"
```

---

### Task 8: End-to-end dry-run test

**Files:**
- Create: `libs/cli/tests/unit_tests/test_deploy_e2e.py`

- [ ] **Step 1: Write an end-to-end dry-run test**

Create `libs/cli/tests/unit_tests/test_deploy_e2e.py`:

```python
"""End-to-end test for deploy bundling with subagents."""

import ast
import json
from pathlib import Path

from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
from deepagents_cli.deploy.config import (
    AgentConfig,
    DeployConfig,
    load_subagents,
)


def _scaffold_project(root: Path) -> None:
    """Create a full project with main agent + 2 subagents."""
    (root / "AGENTS.md").write_text("# Main Agent\nYou are the main agent.\n")
    (root / "deepagents.toml").write_text(
        '[agent]\nname = "my-agent"\nmodel = "anthropic:claude-sonnet-4-6"\n\n'
        '[sandbox]\nprovider = "none"\n'
    )

    # Skill for main agent.
    skill_dir = root / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\n\nReview the code.\n"
    )

    # Subagent: researcher (with skills, no mcp)
    researcher = root / "agents" / "researcher"
    researcher.mkdir(parents=True)
    (researcher / "AGENTS.md").write_text(
        "---\nname: researcher\ndescription: Research topics\n---\n\n"
        "You are a research assistant.\n"
    )
    (researcher / "deepagents.toml").write_text(
        '[agent]\nname = "researcher"\nmodel = "anthropic:claude-haiku-4-5-20251001"\n'
    )
    r_skill = researcher / "skills" / "summarize"
    r_skill.mkdir(parents=True)
    (r_skill / "SKILL.md").write_text(
        "---\nname: summarize\ndescription: Summarize findings\n---\n\nSummarize.\n"
    )

    # Subagent: reviewer (no skills, with mcp)
    reviewer = root / "agents" / "reviewer"
    reviewer.mkdir(parents=True)
    (reviewer / "AGENTS.md").write_text(
        "---\nname: reviewer\ndescription: Review code changes\n---\n\n"
        "You are a code reviewer.\n"
    )
    (reviewer / "deepagents.toml").write_text(
        '[agent]\nname = "reviewer"\nmodel = "anthropic:claude-sonnet-4-6"\n'
    )
    (reviewer / "mcp.json").write_text(
        '{"mcpServers": {"gh": {"type": "http", "url": "http://localhost:3000/github"}}}'
    )


class TestDeployEndToEnd:
    """Full pipeline: scaffold → load config → load subagents → bundle."""

    def test_full_bundle_with_subagents(self, tmp_path: Path) -> None:
        """Bundle a project with two subagents and verify all outputs."""
        project = tmp_path / "project"
        project.mkdir()
        _scaffold_project(project)

        # Load subagents.
        subagents = load_subagents(project)
        assert len(subagents) == 2

        config = DeployConfig(
            agent=AgentConfig(name="my-agent", model="anthropic:claude-sonnet-4-6"),
            subagents=subagents,
        )

        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)

        # Check _seed.json.
        seed = json.loads((build_dir / "_seed.json").read_text())
        assert "researcher" in seed["subagents"]
        assert "reviewer" in seed["subagents"]
        assert len(seed["subagents"]["researcher"]["skills"]) > 0
        assert "/AGENTS.md" in seed["subagents"]["researcher"]["memories"]
        assert "/AGENTS.md" in seed["subagents"]["reviewer"]["memories"]

        # Check subagent MCP file copied.
        assert (build_dir / "_mcp_reviewer.json").is_file()
        assert not (build_dir / "_mcp_researcher.json").exists()

        # Check deploy_graph.py is valid Python and mentions subagents.
        graph_py = (build_dir / "deploy_graph.py").read_text()
        ast.parse(graph_py)
        assert "SUBAGENT_CONFIGS" in graph_py
        assert "researcher" in graph_py
        assert "reviewer" in graph_py
        assert "_load_mcp_tools_reviewer" in graph_py

        # Check pyproject.toml includes MCP dep (reviewer has mcp).
        pyproject = (build_dir / "pyproject.toml").read_text()
        assert "langchain-mcp-adapters" in pyproject

        # Smoke-test print_bundle_summary (should not raise).
        print_bundle_summary(config, build_dir)
```

- [ ] **Step 2: Run the test**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_e2e.py -v
```

Expected: PASS.

- [ ] **Step 3: Run all existing deploy-related tests to check for regressions**

```bash
cd libs/cli && python -m pytest tests/unit_tests/test_deploy_config.py tests/unit_tests/test_deploy_bundler.py tests/unit_tests/test_deploy_e2e.py -v
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/cli/tests/unit_tests/test_deploy_e2e.py
git commit -m "test(cli): add end-to-end deploy bundling test with subagents"
```

---

### Deferred: Per-subagent sandbox overrides in generated graph

The spec describes generating separate `_get_or_create_sandbox_{name}()` functions for subagents that override the parent's sandbox provider. For v1, all subagents inherit the parent's sandbox — the `SubagentConfig.sandbox` field is parsed and stored but the template does not generate per-subagent sandbox blocks. This is a follow-up task once the core subagent wiring is proven out.
