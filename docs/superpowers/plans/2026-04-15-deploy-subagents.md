# Deploy Subagents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sync and async subagent support to deepagents deploy so developers can declare subagents via the filesystem convention and TOML config.

**Architecture:** Extend `config.py` with new dataclasses and parsing for `[[async_subagents]]` and `subagents/` directories. Extend `bundler.py` to include subagent data in `_seed.json`. Extend `templates.py` to generate runtime code that constructs `SubAgent`/`AsyncSubAgent` dicts and passes them to `create_deep_agent()`. Add a GTM agent example exercising both types.

**Tech Stack:** Python 3.12, TOML (tomllib), LangGraph, deepagents SDK

---

### Task 1: Add `description` and `async_subagents` to config parsing

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_config.py`

- [ ] **Step 1: Write failing tests for `description` field on `AgentConfig`**

In `test_config.py`, add to `TestAgentConfig`:

```python
def test_description_default(self) -> None:
    cfg = AgentConfig(name="my-agent")
    assert cfg.description == ""

def test_description_custom(self) -> None:
    cfg = AgentConfig(name="my-agent", description="A helpful bot")
    assert cfg.description == "A helpful bot"
```

Add to `TestParseConfig`:

```python
def test_description_parsed(self) -> None:
    cfg = _parse_config({"agent": {"name": "bot", "description": "A bot"}})
    assert cfg.agent.description == "A bot"

def test_description_optional(self) -> None:
    cfg = _parse_config({"agent": {"name": "bot"}})
    assert cfg.agent.description == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py::TestAgentConfig::test_description_default tests/unit_tests/deploy/test_config.py::TestAgentConfig::test_description_custom tests/unit_tests/deploy/test_config.py::TestParseConfig::test_description_parsed tests/unit_tests/deploy/test_config.py::TestParseConfig::test_description_optional -v`

Expected: FAIL — `description` not a field on `AgentConfig`, and `_parse_config` rejects unknown key `description`.

- [ ] **Step 3: Add `description` to `AgentConfig` and update parsing**

In `config.py`, add `description` field to `AgentConfig`:

```python
@dataclass(frozen=True)
class AgentConfig:
    """`[agent]` section — core agent identity."""

    name: str
    description: str = ""
    model: str = "anthropic:claude-sonnet-4-6"

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "AgentConfig.name must be non-empty"
            raise ValueError(msg)
```

Update `_ALLOWED_AGENT_KEYS`:

```python
_ALLOWED_AGENT_KEYS = frozenset({"name", "description", "model"})
```

Update `_parse_config` to pass `description` through:

```python
agent_kwargs: dict[str, Any] = {"name": agent_data["name"]}
if "description" in agent_data:
    agent_kwargs["description"] = agent_data["description"]
if "model" in agent_data:
    agent_kwargs["model"] = agent_data["model"]
agent = AgentConfig(**agent_kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Write failing tests for `AsyncSubAgentConfig` and `DeployConfig.async_subagents`**

Add new imports at the top of `test_config.py`:

```python
from deepagents_cli.deploy.config import (
    # ... existing imports ...
    AsyncSubAgentConfig,
)
```

Add a new test class:

```python
class TestAsyncSubAgentConfig:
    def test_valid_construction(self) -> None:
        cfg = AsyncSubAgentConfig(
            name="researcher",
            description="Research agent",
            graph_id="research-graph",
        )
        assert cfg.name == "researcher"
        assert cfg.description == "Research agent"
        assert cfg.graph_id == "research-graph"
        assert cfg.url == ""
        assert cfg.headers == {}

    def test_with_url_and_headers(self) -> None:
        cfg = AsyncSubAgentConfig(
            name="r",
            description="d",
            graph_id="g",
            url="https://example.com",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.url == "https://example.com"
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_frozen(self) -> None:
        cfg = AsyncSubAgentConfig(name="r", description="d", graph_id="g")
        with pytest.raises(AttributeError):
            cfg.name = "y"  # type: ignore[misc]
```

Add to `TestParseConfig`:

```python
def test_async_subagents_parsed(self) -> None:
    data: dict[str, Any] = {
        "agent": {"name": "bot"},
        "async_subagents": [
            {
                "name": "researcher",
                "description": "Research agent",
                "graph_id": "research-graph",
                "url": "https://example.com",
            },
        ],
    }
    cfg = _parse_config(data)
    assert len(cfg.async_subagents) == 1
    assert cfg.async_subagents[0].name == "researcher"
    assert cfg.async_subagents[0].graph_id == "research-graph"
    assert cfg.async_subagents[0].url == "https://example.com"

def test_async_subagents_defaults_empty(self) -> None:
    cfg = _parse_config({"agent": {"name": "bot"}})
    assert cfg.async_subagents == []

def test_async_subagents_missing_required_field(self) -> None:
    data: dict[str, Any] = {
        "agent": {"name": "bot"},
        "async_subagents": [{"name": "r", "description": "d"}],
    }
    with pytest.raises(ValueError, match="graph_id.*required"):
        _parse_config(data)

def test_async_subagents_unknown_key(self) -> None:
    data: dict[str, Any] = {
        "agent": {"name": "bot"},
        "async_subagents": [
            {"name": "r", "description": "d", "graph_id": "g", "bogus": "x"},
        ],
    }
    with pytest.raises(ValueError, match="Unknown key"):
        _parse_config(data)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py::TestAsyncSubAgentConfig tests/unit_tests/deploy/test_config.py::TestParseConfig::test_async_subagents_parsed tests/unit_tests/deploy/test_config.py::TestParseConfig::test_async_subagents_defaults_empty tests/unit_tests/deploy/test_config.py::TestParseConfig::test_async_subagents_missing_required_field tests/unit_tests/deploy/test_config.py::TestParseConfig::test_async_subagents_unknown_key -v`

Expected: FAIL — `AsyncSubAgentConfig` doesn't exist yet.

- [ ] **Step 7: Implement `AsyncSubAgentConfig` and update `DeployConfig` and `_parse_config`**

In `config.py`, add the new dataclass after `SandboxConfig`:

```python
_ALLOWED_ASYNC_SUBAGENT_KEYS = frozenset({"name", "description", "graph_id", "url", "headers"})


@dataclass(frozen=True)
class AsyncSubAgentConfig:
    """An async subagent referencing a remote deployed agent."""

    name: str
    description: str
    graph_id: str
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
```

Update `DeployConfig`:

```python
@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from `deepagents.toml`."""

    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    async_subagents: list[AsyncSubAgentConfig] = field(default_factory=list)
```

Update `_ALLOWED_SECTIONS`:

```python
_ALLOWED_SECTIONS = frozenset({"agent", "sandbox", "async_subagents"})
```

Update `_parse_config` — add async subagent parsing after the sandbox parsing block:

```python
async_subagent_data = data.get("async_subagents", [])
async_subagents: list[AsyncSubAgentConfig] = []
for i, entry in enumerate(async_subagent_data):
    unknown_async = set(entry.keys()) - _ALLOWED_ASYNC_SUBAGENT_KEYS
    if unknown_async:
        msg = (
            f"Unknown key(s) in [[async_subagents]][{i}]: {sorted(unknown_async)}. "
            f"Allowed: {sorted(_ALLOWED_ASYNC_SUBAGENT_KEYS)}"
        )
        raise ValueError(msg)
    for required in ("name", "description", "graph_id"):
        if required not in entry:
            msg = f"[[async_subagents]][{i}].{required} is required"
            raise ValueError(msg)
    async_kwargs: dict[str, Any] = {
        k: entry[k] for k in _ALLOWED_ASYNC_SUBAGENT_KEYS if k in entry
    }
    async_subagents.append(AsyncSubAgentConfig(**async_kwargs))

return DeployConfig(agent=agent, sandbox=sandbox, async_subagents=async_subagents)
```

- [ ] **Step 8: Run all config tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py -v`
Expected: All PASS.

- [ ] **Step 9: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py libs/cli/tests/unit_tests/deploy/test_config.py
git commit -m "feat(deploy): add description to AgentConfig and async_subagents to DeployConfig"
```

---

### Task 2: Add sync subagent config loading and validation

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_config.py`

- [ ] **Step 1: Write failing tests for subagent config loading**

Add new imports at top of `test_config.py`:

```python
from deepagents_cli.deploy.config import (
    # ... existing imports ...
    SubAgentConfig,
    SubAgentProject,
    load_subagents,
    SUBAGENTS_DIRNAME,
)
```

Add a new test class:

```python
class TestLoadSubagents:
    def _make_subagent(
        self, parent: Path, name: str, *, description: str = "A subagent"
    ) -> Path:
        """Create a minimal subagent directory."""
        sa_dir = parent / SUBAGENTS_DIRNAME / name
        sa_dir.mkdir(parents=True, exist_ok=True)
        toml_content = f'[agent]\nname = "{name}"\ndescription = "{description}"\n'
        (sa_dir / "deepagents.toml").write_text(toml_content, encoding="utf-8")
        (sa_dir / "AGENTS.md").write_text(f"# {name}", encoding="utf-8")
        return sa_dir

    def test_no_subagents_dir(self, tmp_path: Path) -> None:
        result = load_subagents(tmp_path)
        assert result == {}

    def test_empty_subagents_dir(self, tmp_path: Path) -> None:
        (tmp_path / SUBAGENTS_DIRNAME).mkdir()
        result = load_subagents(tmp_path)
        assert result == {}

    def test_single_subagent(self, tmp_path: Path) -> None:
        self._make_subagent(tmp_path, "researcher")
        result = load_subagents(tmp_path)
        assert "researcher" in result
        assert result["researcher"].config.agent.name == "researcher"
        assert result["researcher"].config.agent.description == "A subagent"

    def test_multiple_subagents(self, tmp_path: Path) -> None:
        self._make_subagent(tmp_path, "researcher")
        self._make_subagent(tmp_path, "coder")
        result = load_subagents(tmp_path)
        assert len(result) == 2
        assert "researcher" in result
        assert "coder" in result

    def test_missing_agents_md_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sa_dir.mkdir(parents=True)
        toml = '[agent]\nname = "bad"\ndescription = "d"\n'
        (sa_dir / "deepagents.toml").write_text(toml, encoding="utf-8")
        with pytest.raises(ValueError, match="AGENTS.md.*required"):
            load_subagents(tmp_path)

    def test_missing_toml_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text("# bad", encoding="utf-8")
        with pytest.raises(ValueError, match="deepagents.toml.*required"):
            load_subagents(tmp_path)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sa_dir.mkdir(parents=True)
        (sa_dir / "deepagents.toml").write_text(
            '[agent]\nname = "bad"\n', encoding="utf-8"
        )
        (sa_dir / "AGENTS.md").write_text("# bad", encoding="utf-8")
        with pytest.raises(ValueError, match="description.*required"):
            load_subagents(tmp_path)

    def test_sandbox_section_rejected(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sa_dir.mkdir(parents=True)
        toml = '[agent]\nname = "bad"\ndescription = "d"\n\n[sandbox]\nprovider = "none"\n'
        (sa_dir / "deepagents.toml").write_text(toml, encoding="utf-8")
        (sa_dir / "AGENTS.md").write_text("# bad", encoding="utf-8")
        with pytest.raises(ValueError, match="sandbox.*not allowed"):
            load_subagents(tmp_path)

    def test_async_subagents_section_rejected(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sa_dir.mkdir(parents=True)
        toml = (
            '[agent]\nname = "bad"\ndescription = "d"\n\n'
            '[[async_subagents]]\nname = "x"\ndescription = "x"\ngraph_id = "x"\n'
        )
        (sa_dir / "deepagents.toml").write_text(toml, encoding="utf-8")
        (sa_dir / "AGENTS.md").write_text("# bad", encoding="utf-8")
        with pytest.raises(ValueError, match="async_subagents.*not allowed"):
            load_subagents(tmp_path)

    def test_nested_subagents_rejected(self, tmp_path: Path) -> None:
        sa_dir = self._make_subagent(tmp_path, "parent-sa")
        (sa_dir / SUBAGENTS_DIRNAME).mkdir()
        with pytest.raises(ValueError, match="Nested.*not allowed"):
            load_subagents(tmp_path)

    def test_subagent_with_skills(self, tmp_path: Path) -> None:
        sa_dir = self._make_subagent(tmp_path, "researcher")
        skill_dir = sa_dir / "skills" / "search"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Search", encoding="utf-8")
        result = load_subagents(tmp_path)
        assert result["researcher"].root == sa_dir

    def test_subagent_mcp_validated(self, tmp_path: Path) -> None:
        sa_dir = self._make_subagent(tmp_path, "researcher")
        mcp = {"mcpServers": {"s": {"type": "stdio", "command": "node"}}}
        (sa_dir / "mcp.json").write_text(json.dumps(mcp), encoding="utf-8")
        with pytest.raises(ValueError, match="stdio"):
            load_subagents(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py::TestLoadSubagents -v`
Expected: FAIL — `SubAgentConfig`, `SubAgentProject`, `load_subagents`, `SUBAGENTS_DIRNAME` don't exist.

- [ ] **Step 3: Implement `SubAgentConfig`, `SubAgentProject`, and `load_subagents`**

In `config.py`, add the constant near the other filename constants:

```python
SUBAGENTS_DIRNAME = "subagents"
```

Add the new dataclasses after `AsyncSubAgentConfig`:

```python
@dataclass(frozen=True)
class SubAgentConfig:
    """Parsed from a subagent's deepagents.toml."""

    agent: AgentConfig


@dataclass(frozen=True)
class SubAgentProject:
    """A discovered subagent directory with its parsed config."""

    config: SubAgentConfig
    root: Path
```

Add the allowed sections for subagent TOML:

```python
_ALLOWED_SUBAGENT_SECTIONS = frozenset({"agent"})
```

Add `_parse_subagent_config` function:

```python
def _parse_subagent_config(data: dict[str, Any], subagent_dir: str) -> SubAgentConfig:
    """Parse a subagent's deepagents.toml."""
    for section in data:
        if section not in _ALLOWED_SUBAGENT_SECTIONS:
            msg = (
                f"Subagent '{subagent_dir}': [{section}] is not allowed in "
                f"subagent config. Only {sorted(_ALLOWED_SUBAGENT_SECTIONS)} "
                f"are allowed."
            )
            raise ValueError(msg)

    agent_data = data.get("agent", {})
    if "name" not in agent_data:
        msg = f"Subagent '{subagent_dir}': [agent].name is required"
        raise ValueError(msg)
    if not agent_data.get("description", "").strip():
        msg = f"Subagent '{subagent_dir}': [agent].description is required"
        raise ValueError(msg)

    unknown_agent = set(agent_data.keys()) - _ALLOWED_AGENT_KEYS
    if unknown_agent:
        msg = (
            f"Subagent '{subagent_dir}': Unknown key(s) in [agent]: "
            f"{sorted(unknown_agent)}. Allowed: {sorted(_ALLOWED_AGENT_KEYS)}"
        )
        raise ValueError(msg)

    agent_kwargs: dict[str, Any] = {
        "name": agent_data["name"],
        "description": agent_data["description"],
    }
    if "model" in agent_data:
        agent_kwargs["model"] = agent_data["model"]
    agent = AgentConfig(**agent_kwargs)

    return SubAgentConfig(agent=agent)
```

Add `load_subagents` function:

```python
def load_subagents(project_root: Path) -> dict[str, SubAgentProject]:
    """Discover and validate subagent directories under ``subagents/``.

    Returns:
        Dict mapping subagent name to its parsed config and root path.

    Raises:
        ValueError: If any subagent directory is invalid.
    """
    subagents_dir = project_root / SUBAGENTS_DIRNAME
    if not subagents_dir.is_dir():
        return {}

    result: dict[str, SubAgentProject] = {}
    for entry in sorted(subagents_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        # Reject nested subagents/ directories.
        if (entry / SUBAGENTS_DIRNAME).exists():
            msg = (
                f"Subagent '{entry.name}': Nested subagents/ directory is not "
                f"allowed. Only one level of subagents is supported."
            )
            raise ValueError(msg)

        # Require deepagents.toml.
        toml_path = entry / DEFAULT_CONFIG_FILENAME
        if not toml_path.is_file():
            msg = (
                f"Subagent '{entry.name}': deepagents.toml is required "
                f"in {entry}"
            )
            raise ValueError(msg)

        # Require AGENTS.md.
        agents_md = entry / AGENTS_MD_FILENAME
        if not agents_md.is_file():
            msg = (
                f"Subagent '{entry.name}': AGENTS.md is required in {entry}"
            )
            raise ValueError(msg)

        # Parse and validate the subagent config.
        try:
            with toml_path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Subagent '{entry.name}': Syntax error in deepagents.toml: {exc}"
            raise ValueError(msg) from exc

        config = _parse_subagent_config(data, entry.name)

        # Validate MCP if present.
        mcp_path = entry / MCP_FILENAME
        if mcp_path.is_file():
            errors = _validate_mcp_for_deploy(mcp_path)
            if errors:
                msg = f"Subagent '{entry.name}': {'; '.join(errors)}"
                raise ValueError(msg)

        result[config.agent.name] = SubAgentProject(config=config, root=entry)

    return result
```

- [ ] **Step 4: Run all config tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py libs/cli/tests/unit_tests/deploy/test_config.py
git commit -m "feat(deploy): add sync subagent config loading and validation"
```

---

### Task 3: Add subagent name uniqueness validation

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_config.py`

- [ ] **Step 1: Write failing test**

Add to `test_config.py`:

```python
from deepagents_cli.deploy.config import (
    # ... existing imports ...
    validate_subagent_names,
)
```

```python
class TestValidateSubagentNames:
    def test_no_conflict(self) -> None:
        async_subs = [
            AsyncSubAgentConfig(name="a", description="d", graph_id="g"),
        ]
        sync_subs = {
            "b": SubAgentProject(
                config=SubAgentConfig(
                    agent=AgentConfig(name="b", description="d"),
                ),
                root=Path("/fake"),
            ),
        }
        errors = validate_subagent_names(async_subs, sync_subs)
        assert errors == []

    def test_duplicate_across_sync_and_async(self) -> None:
        async_subs = [
            AsyncSubAgentConfig(name="researcher", description="d", graph_id="g"),
        ]
        sync_subs = {
            "researcher": SubAgentProject(
                config=SubAgentConfig(
                    agent=AgentConfig(name="researcher", description="d"),
                ),
                root=Path("/fake"),
            ),
        }
        errors = validate_subagent_names(async_subs, sync_subs)
        assert len(errors) == 1
        assert "researcher" in errors[0]

    def test_duplicate_async_names(self) -> None:
        async_subs = [
            AsyncSubAgentConfig(name="r", description="d", graph_id="g1"),
            AsyncSubAgentConfig(name="r", description="d", graph_id="g2"),
        ]
        errors = validate_subagent_names(async_subs, {})
        assert len(errors) == 1
        assert "r" in errors[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py::TestValidateSubagentNames -v`
Expected: FAIL — `validate_subagent_names` doesn't exist.

- [ ] **Step 3: Implement `validate_subagent_names`**

In `config.py`:

```python
def validate_subagent_names(
    async_subagents: list[AsyncSubAgentConfig],
    sync_subagents: dict[str, SubAgentProject],
) -> list[str]:
    """Check that all subagent names are unique across sync and async.

    Returns:
        List of validation error strings. Empty if all names are unique.
    """
    errors: list[str] = []
    seen: dict[str, str] = {}  # name -> source

    for asa in async_subagents:
        if asa.name in seen:
            errors.append(
                f"Duplicate subagent name '{asa.name}': declared in both "
                f"{seen[asa.name]} and async_subagents"
            )
        else:
            seen[asa.name] = "async_subagents"

    for name in sync_subagents:
        if name in seen:
            errors.append(
                f"Duplicate subagent name '{name}': declared in both "
                f"{seen[name]} and subagents/"
            )
        else:
            seen[name] = "subagents/"

    return errors
```

- [ ] **Step 4: Run tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py libs/cli/tests/unit_tests/deploy/test_config.py
git commit -m "feat(deploy): add subagent name uniqueness validation"
```

---

### Task 4: Extend bundler to include subagents in `_seed.json`

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_bundler.py`

- [ ] **Step 1: Write failing tests for subagent seed building**

Add new imports to `test_bundler.py`:

```python
from deepagents_cli.deploy.config import (
    # ... existing imports ...
    AsyncSubAgentConfig,
    SUBAGENTS_DIRNAME,
)
```

Add a helper to create subagent directories:

```python
def _add_subagent(
    project: Path,
    name: str,
    *,
    description: str = "A subagent",
    skills: dict[str, str] | None = None,
    mcp: dict | None = None,
) -> Path:
    """Add a subagent directory to an existing project."""
    sa_dir = project / SUBAGENTS_DIRNAME / name
    sa_dir.mkdir(parents=True, exist_ok=True)
    toml = f'[agent]\nname = "{name}"\ndescription = "{description}"\n'
    (sa_dir / "deepagents.toml").write_text(toml, encoding="utf-8")
    (sa_dir / "AGENTS.md").write_text(f"# {name} prompt", encoding="utf-8")
    if skills:
        for skill_path, content in skills.items():
            skill_file = sa_dir / "skills" / skill_path
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(content, encoding="utf-8")
    if mcp is not None:
        (sa_dir / "mcp.json").write_text(json.dumps(mcp), encoding="utf-8")
    return sa_dir
```

Add new test class:

```python
class TestBuildSeedSubagents:
    def test_no_subagents_key_when_none(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "subagents" not in seed

    def test_sync_subagent_included(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "researcher", description="Research agent")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "subagents" in seed
        assert "researcher" in seed["subagents"]
        sa = seed["subagents"]["researcher"]
        assert sa["config"]["name"] == "researcher"
        assert sa["config"]["description"] == "Research agent"
        assert sa["memories"]["/AGENTS.md"] == "# researcher prompt"

    def test_sync_subagent_with_skills(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(
            project,
            "researcher",
            skills={"search/SKILL.md": "# Search skill"},
        )
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["researcher"]
        assert "/search/SKILL.md" in sa["skills"]
        assert sa["skills"]["/search/SKILL.md"] == "# Search skill"

    def test_sync_subagent_with_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        mcp_data = {"mcpServers": {"s": {"type": "http", "url": "http://x"}}}
        _add_subagent(project, "researcher", mcp=mcp_data)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["researcher"]
        assert sa["mcp"] == mcp_data

    def test_sync_subagent_no_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "researcher")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["researcher"]
        assert sa["mcp"] is None

    def test_async_subagents_included(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            sandbox=SandboxConfig(),
            async_subagents=[
                AsyncSubAgentConfig(
                    name="writer",
                    description="Content writer",
                    graph_id="writer-graph",
                    url="https://example.com",
                ),
            ],
        )
        seed = _build_seed(config, project, "# prompt")
        assert "async_subagents" in seed
        assert len(seed["async_subagents"]) == 1
        assert seed["async_subagents"][0]["name"] == "writer"
        assert seed["async_subagents"][0]["graph_id"] == "writer-graph"

    def test_no_async_subagents_key_when_none(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "async_subagents" not in seed

    def test_multiple_sync_subagents(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "researcher")
        _add_subagent(project, "coder")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert len(seed["subagents"]) == 2
        assert "researcher" in seed["subagents"]
        assert "coder" in seed["subagents"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py::TestBuildSeedSubagents -v`
Expected: FAIL — `_build_seed` doesn't handle subagents yet.

- [ ] **Step 3: Implement subagent seed building in bundler**

In `bundler.py`, add imports:

```python
from deepagents_cli.deploy.config import (
    # ... existing imports ...
    SUBAGENTS_DIRNAME,
    SubAgentProject,
    load_subagents,
)
```

Add `_build_subagent_seed` function:

```python
def _build_subagent_seed(subagent: SubAgentProject) -> dict:
    """Build the seed entry for a single sync subagent.

    Returns a dict with keys: config, memories, skills, mcp.
    """
    sa_root = subagent.root
    agent = subagent.config.agent

    # Read AGENTS.md (required — already validated).
    agents_md = sa_root / AGENTS_MD_FILENAME
    memories: dict[str, str] = {
        f"/{AGENTS_MD_FILENAME}": agents_md.read_text(encoding="utf-8"),
    }

    # Discover skills.
    skills: dict[str, str] = {}
    skills_dir = sa_root / SKILLS_DIRNAME
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skills_dir).as_posix()
                skills[f"/{rel}"] = f.read_text(encoding="utf-8")

    # Read MCP config if present.
    mcp_path = sa_root / MCP_FILENAME
    mcp = None
    if mcp_path.is_file():
        mcp = json.loads(mcp_path.read_text(encoding="utf-8"))

    return {
        "config": {
            "name": agent.name,
            "description": agent.description,
            "model": agent.model,
        },
        "memories": memories,
        "skills": skills,
        "mcp": mcp,
    }
```

Update `_build_seed` to accept `config` and use it — remove the `# noqa: ARG001` on the `config` parameter and add subagent logic:

```python
def _build_seed(
    config: DeployConfig,
    project_root: Path,
    system_prompt: str,
) -> dict:
    # ... existing memories/skills/user_memories logic stays the same ...

    seed: dict[str, Any] = {
        "memories": memories,
        "skills": skills,
        "user_memories": user_memories,
    }

    # Sync subagents.
    sync_subagents = load_subagents(project_root)
    if sync_subagents:
        seed["subagents"] = {
            name: _build_subagent_seed(sa)
            for name, sa in sync_subagents.items()
        }

    # Async subagents.
    if config.async_subagents:
        seed["async_subagents"] = [
            {
                "name": asa.name,
                "description": asa.description,
                "graph_id": asa.graph_id,
                "url": asa.url,
            }
            for asa in config.async_subagents
        ]

    return seed
```

Note: The return statement replaces the old `return {"memories": ..., "skills": ..., "user_memories": ...}`. The existing `user_memories` logic stays the same but is folded into the `seed` dict that gets conditionally extended.

- [ ] **Step 4: Run all bundler tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/deploy/test_bundler.py
git commit -m "feat(deploy): extend bundler to include subagents in _seed.json"
```

---

### Task 5: Extend pyproject.toml dependency inference for subagents

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_bundler.py`

- [ ] **Step 1: Write failing tests**

Add to `TestRenderPyproject` in `test_bundler.py`:

```python
def test_subagent_model_dep_inferred(self) -> None:
    """Subagent with a different model provider adds its dependency."""
    config = DeployConfig(
        agent=AgentConfig(name="test-agent", model="anthropic:claude-sonnet-4-6"),
        sandbox=SandboxConfig(),
    )
    # We need to pass subagent model providers to the render function.
    result = _render_pyproject(
        config,
        mcp_present=False,
        subagent_model_providers=["openai"],
    )
    assert "langchain-openai" in result

def test_subagent_mcp_adds_dep(self) -> None:
    """Any subagent with MCP should trigger langchain-mcp-adapters."""
    config = _minimal_config()
    result = _render_pyproject(
        config,
        mcp_present=False,
        has_subagent_mcp=True,
    )
    assert "langchain-mcp-adapters" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py::TestRenderPyproject::test_subagent_model_dep_inferred tests/unit_tests/deploy/test_bundler.py::TestRenderPyproject::test_subagent_mcp_adds_dep -v`
Expected: FAIL — `_render_pyproject` doesn't accept the new parameters.

- [ ] **Step 3: Update `_render_pyproject` and `bundle`**

In `bundler.py`, update `_render_pyproject`:

```python
def _render_pyproject(
    config: DeployConfig,
    *,
    mcp_present: bool,
    subagent_model_providers: list[str] | None = None,
    has_subagent_mcp: bool = False,
) -> str:
    """Render the deployment package's `pyproject.toml`."""
    deps: list[str] = []

    provider_prefix = (
        config.agent.model.split(":", 1)[0] if ":" in config.agent.model else ""
    )
    if provider_prefix and provider_prefix in _MODEL_PROVIDER_DEPS:
        deps.append(_MODEL_PROVIDER_DEPS[provider_prefix])

    # Add deps for subagent model providers.
    for sp in subagent_model_providers or []:
        dep = _MODEL_PROVIDER_DEPS.get(sp)
        if dep and dep not in deps:
            deps.append(dep)

    if mcp_present or has_subagent_mcp:
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

Update the `bundle` function to compute and pass subagent info:

```python
# In bundle(), after building the seed:
sync_subagents = load_subagents(project_root)
subagent_model_providers: list[str] = []
has_subagent_mcp = False
for sa in sync_subagents.values():
    model = sa.config.agent.model
    if ":" in model:
        subagent_model_providers.append(model.split(":", 1)[0])
    if (sa.root / MCP_FILENAME).is_file():
        has_subagent_mcp = True

# Then pass to _render_pyproject:
(build_dir / "pyproject.toml").write_text(
    _render_pyproject(
        config,
        mcp_present=mcp_present,
        subagent_model_providers=subagent_model_providers,
        has_subagent_mcp=has_subagent_mcp,
    ),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run all bundler tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/deploy/test_bundler.py
git commit -m "feat(deploy): infer subagent dependencies in pyproject.toml"
```

---

### Task 6: Extend `deploy_graph.py` template for subagents

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/templates.py`
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_bundler.py`

- [ ] **Step 1: Write failing tests**

Add to `TestRenderDeployGraph` in `test_bundler.py`:

```python
def test_subagent_imports_when_sync(self) -> None:
    config = _minimal_config()
    result = _render_deploy_graph(
        config,
        mcp_present=False,
        has_sync_subagents=True,
    )
    compile(result, "<deploy_graph_sync_sa>", "exec")
    assert "from deepagents.middleware.subagents import SubAgent" in result
    assert "_build_sync_subagents" in result

def test_subagent_imports_when_async(self) -> None:
    config = _minimal_config()
    result = _render_deploy_graph(
        config,
        mcp_present=False,
        has_async_subagents=True,
    )
    compile(result, "<deploy_graph_async_sa>", "exec")
    assert "from deepagents.middleware.async_subagents import AsyncSubAgent" in result
    assert "_build_async_subagents" in result

def test_no_subagent_imports_when_none(self) -> None:
    config = _minimal_config()
    result = _render_deploy_graph(config, mcp_present=False)
    assert "SubAgent" not in result
    assert "AsyncSubAgent" not in result

def test_subagents_passed_to_create_deep_agent(self) -> None:
    config = _minimal_config()
    result = _render_deploy_graph(
        config,
        mcp_present=False,
        has_sync_subagents=True,
        has_async_subagents=True,
    )
    compile(result, "<deploy_graph_both_sa>", "exec")
    assert "subagents=" in result

def test_subagent_seeding(self) -> None:
    config = _minimal_config()
    result = _render_deploy_graph(
        config,
        mcp_present=False,
        has_sync_subagents=True,
    )
    assert "_seed_subagents_if_needed" in result
    assert '"subagents"' in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py::TestRenderDeployGraph::test_subagent_imports_when_sync tests/unit_tests/deploy/test_bundler.py::TestRenderDeployGraph::test_subagent_imports_when_async tests/unit_tests/deploy/test_bundler.py::TestRenderDeployGraph::test_no_subagent_imports_when_none tests/unit_tests/deploy/test_bundler.py::TestRenderDeployGraph::test_subagents_passed_to_create_deep_agent tests/unit_tests/deploy/test_bundler.py::TestRenderDeployGraph::test_subagent_seeding -v`
Expected: FAIL — `_render_deploy_graph` doesn't accept `has_sync_subagents` / `has_async_subagents`.

- [ ] **Step 3: Add subagent template blocks to `templates.py`**

In `templates.py`, add after the MCP template block:

```python
# ---------------------------------------------------------------------------
# Subagent builder blocks (emitted conditionally)
# ---------------------------------------------------------------------------

SYNC_SUBAGENTS_TEMPLATE = '''\
from deepagents.middleware.subagents import SubAgent


async def _build_sync_subagents(seed, store, assistant_id):
    """Build SubAgent dicts from seed data and seed their memories/skills."""
    subagents_data = seed.get("subagents", {})
    if not subagents_data:
        return []

    subagents = []
    for name, data in subagents_data.items():
        sa: SubAgent = {
            "name": data["config"]["name"],
            "description": data["config"]["description"],
            "system_prompt": data["memories"]["/AGENTS.md"],
        }
        if data["config"].get("model"):
            sa["model"] = data["config"]["model"]

        # Seed subagent memories and skills into store under subagent namespace.
        sa_ns = (assistant_id, "subagents", name)
        if store is not None:
            for path, content in data.get("memories", {}).items():
                if await store.aget(sa_ns, path) is None:
                    await store.aput(
                        sa_ns,
                        path,
                        {"content": content, "encoding": "utf-8"},
                    )
            for path, content in data.get("skills", {}).items():
                if await store.aget(sa_ns, path) is None:
                    await store.aput(
                        sa_ns,
                        path,
                        {"content": content, "encoding": "utf-8"},
                    )

        if data.get("skills"):
            sa_skills_prefix = f"/memories/subagents/{name}/skills/"
            sa["skills"] = [sa_skills_prefix]

        if data.get("mcp"):
            sa["tools"] = await _load_subagent_mcp_tools(data["mcp"])

        subagents.append(sa)
    return subagents


async def _load_subagent_mcp_tools(mcp_config):
    """Load MCP tools for a subagent from its mcp config."""
    servers = mcp_config.get("mcpServers", {})
    connections = {}
    for name, cfg in servers.items():
        transport = cfg.get("type", cfg.get("transport", "stdio"))
        if transport in ("http", "sse"):
            conn = {"transport": transport, "url": cfg["url"]}
            if "headers" in cfg:
                conn["headers"] = cfg["headers"]
            connections[name] = conn

    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(connections)
        return await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load subagent MCP tools: %s", exc)
        return []
'''

ASYNC_SUBAGENTS_TEMPLATE = '''\
from deepagents.middleware.async_subagents import AsyncSubAgent


def _build_async_subagents(seed):
    """Build AsyncSubAgent dicts from seed data."""
    async_data = seed.get("async_subagents", [])
    if not async_data:
        return []

    subagents = []
    for entry in async_data:
        asa: AsyncSubAgent = {
            "name": entry["name"],
            "description": entry["description"],
            "graph_id": entry["graph_id"],
        }
        if entry.get("url"):
            asa["url"] = entry["url"]
        if entry.get("headers"):
            asa["headers"] = entry["headers"]
        subagents.append(asa)
    return subagents
'''
```

- [ ] **Step 4: Update `DEPLOY_GRAPH_TEMPLATE` for subagents**

In `templates.py`, update the template to accept subagent blocks. Add new format placeholders:

In the template, after `{mcp_tools_block}`, add:

```
{sync_subagents_block}

{async_subagents_block}
```

In the `make_graph` function body in the template, after the MCP tools loading line, add:

```python
    all_subagents: list = []
    {sync_subagents_load_call}
    {async_subagents_load_call}
```

Update the `create_deep_agent` call to pass subagents:

```python
    return create_deep_agent(
        model={model!r},
        memory=memory_sources,
        skills=[SKILLS_PREFIX],
        tools=tools,
        backend=backend_factory,
        permissions=permissions,
        middleware=[
            SandboxSyncMiddleware(backend=backend_factory, sources=[SKILLS_PREFIX]),
        ],
        subagents=all_subagents or None,
    )
```

Also add subagent store routes to `_build_backend_factory` — for each sync subagent, there needs to be a store route. Since subagent names aren't known at template time (they're in the seed), add a dynamic route builder:

In `_build_backend_factory`, after the existing routes:

```python
        # Add subagent store routes for seeded sync subagents.
        seed = _load_seed()
        for sa_name in seed.get("subagents", {{}}):
            sa_prefix = f"{{MEMORIES_PREFIX}}subagents/{sa_name}/"
            routes[sa_prefix] = StoreBackend(
                namespace=_make_namespace_factory(assistant_id, "subagents", sa_name),
            )
```

- [ ] **Step 5: Update `_render_deploy_graph` in `bundler.py`**

```python
def _render_deploy_graph(
    config: DeployConfig,
    *,
    mcp_present: bool,
    has_user_memories: bool = False,
    has_sync_subagents: bool = False,
    has_async_subagents: bool = False,
) -> str:
    """Render the generated `deploy_graph.py`."""
    # ... existing provider/mcp logic ...

    if has_sync_subagents:
        sync_subagents_block = SYNC_SUBAGENTS_TEMPLATE
        sync_subagents_load_call = (
            "all_subagents.extend(await _build_sync_subagents(seed, store, assistant_id))"
        )
    else:
        sync_subagents_block = ""
        sync_subagents_load_call = "pass  # no sync subagents"

    if has_async_subagents:
        async_subagents_block = ASYNC_SUBAGENTS_TEMPLATE
        async_subagents_load_call = (
            "all_subagents.extend(_build_async_subagents(seed))"
        )
    else:
        async_subagents_block = ""
        async_subagents_load_call = "pass  # no async subagents"

    return DEPLOY_GRAPH_TEMPLATE.format(
        # ... existing format args ...
        sync_subagents_block=sync_subagents_block,
        async_subagents_block=async_subagents_block,
        sync_subagents_load_call=sync_subagents_load_call,
        async_subagents_load_call=async_subagents_load_call,
    )
```

Also update `bundle()` to pass the new flags:

```python
has_sync_subagents = bool(sync_subagents)
has_async_subagents = bool(config.async_subagents)

(build_dir / "deploy_graph.py").write_text(
    _render_deploy_graph(
        config,
        mcp_present=mcp_present,
        has_user_memories=has_user_memories,
        has_sync_subagents=has_sync_subagents,
        has_async_subagents=has_async_subagents,
    ),
    encoding="utf-8",
)
```

- [ ] **Step 6: Run all bundler tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py -v`
Expected: All PASS.

- [ ] **Step 7: Verify generated code is valid Python**

The existing `test_output_is_valid_python` and `test_each_provider_renders` tests use `compile()` to verify the generated code. The new tests also use `compile()`. Run the full test suite to confirm.

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/ -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/templates.py libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/deploy/test_bundler.py
git commit -m "feat(deploy): extend template and bundler for subagent runtime construction"
```

---

### Task 7: Update `print_bundle_summary` for subagents

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Test: `libs/cli/tests/unit_tests/deploy/test_bundler.py`

- [ ] **Step 1: Write failing tests**

Add to `TestPrintBundleSummary` in `test_bundler.py`:

```python
def test_sync_subagent_summary(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = {
        "memories": {"/AGENTS.md": "x"},
        "skills": {},
        "subagents": {
            "researcher": {
                "config": {"name": "researcher", "description": "d", "model": "anthropic:claude-sonnet-4-6"},
                "memories": {"/AGENTS.md": "y"},
                "skills": {"/search/SKILL.md": "z"},
                "mcp": None,
            },
        },
    }
    (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
    config = _minimal_config()
    print_bundle_summary(config, tmp_path)
    out = capsys.readouterr().out
    assert "Subagents (1 sync" in out
    assert "researcher" in out

def test_async_subagent_summary(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = {
        "memories": {"/AGENTS.md": "x"},
        "skills": {},
        "async_subagents": [
            {"name": "writer", "description": "d", "graph_id": "g", "url": ""},
        ],
    }
    (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
    config = _minimal_config()
    print_bundle_summary(config, tmp_path)
    out = capsys.readouterr().out
    assert "1 async" in out
    assert "writer" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py::TestPrintBundleSummary::test_sync_subagent_summary tests/unit_tests/deploy/test_bundler.py::TestPrintBundleSummary::test_async_subagent_summary -v`
Expected: FAIL — `print_bundle_summary` doesn't print subagent info.

- [ ] **Step 3: Update `print_bundle_summary`**

In `bundler.py`, add to `print_bundle_summary` after the MCP config section:

```python
# Subagent summary.
sync_subagents = seed.get("subagents", {})
async_subagents = seed.get("async_subagents", [])
if sync_subagents or async_subagents:
    parts = []
    if sync_subagents:
        parts.append(f"{len(sync_subagents)} sync")
    if async_subagents:
        parts.append(f"{len(async_subagents)} async")
    print(f"\n  Subagents ({', '.join(parts)}):")
    for name, sa_data in sync_subagents.items():
        desc = sa_data.get("config", {}).get("description", "")
        print(f"    {name} (sync) — {desc}")
    for asa in async_subagents:
        desc = asa.get("description", "")
        print(f"    {asa['name']} (async) — {desc}")
```

- [ ] **Step 4: Run all tests**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py libs/cli/tests/unit_tests/deploy/test_bundler.py
git commit -m "feat(deploy): add subagent info to bundle summary"
```

---

### Task 8: Add GTM agent example

**Files:**
- Create: `examples/deploy-gtm-agent/deepagents.toml`
- Create: `examples/deploy-gtm-agent/AGENTS.md`
- Create: `examples/deploy-gtm-agent/mcp.json`
- Create: `examples/deploy-gtm-agent/skills/competitor-analysis/SKILL.md`
- Create: `examples/deploy-gtm-agent/subagents/market-researcher/deepagents.toml`
- Create: `examples/deploy-gtm-agent/subagents/market-researcher/AGENTS.md`
- Create: `examples/deploy-gtm-agent/subagents/market-researcher/skills/analyze-market/SKILL.md`

- [ ] **Step 1: Create parent agent config**

Create `examples/deploy-gtm-agent/deepagents.toml`:

```toml
[agent]
name = "gtm-agent"
description = "Go-to-market strategy agent that coordinates research and content creation"
model = "anthropic:claude-sonnet-4-6"

[sandbox]
provider = "none"

[[async_subagents]]
name = "content-writer"
description = "Writes blog posts, landing pages, and marketing copy based on research findings"
graph_id = "content-writer-agent"
url = "https://my-langgraph-deployment.com"
```

- [ ] **Step 2: Create parent agent prompt**

Create `examples/deploy-gtm-agent/AGENTS.md`:

```markdown
# GTM Strategy Agent

You are a go-to-market strategy agent that helps teams plan and execute product launches.

## Capabilities

You coordinate between specialized subagents:

- **market-researcher** (sync): Delegates market research tasks — competitor analysis, TAM/SAM/SOM estimation, and audience segmentation.
- **content-writer** (async): Kicks off long-running content creation tasks — blog posts, landing pages, and marketing copy.

## Workflow

1. When given a product or feature to launch, start by delegating market research to the market-researcher subagent.
2. Use the research findings to develop a GTM strategy covering positioning, pricing, and channel selection.
3. Kick off content creation tasks via the content-writer async subagent for any required marketing materials.
4. Monitor async tasks and integrate deliverables into the final GTM plan.

## Guidelines

- Always ground recommendations in research data from the market-researcher.
- Present strategies with clear rationale and supporting evidence.
- When creating content briefs for the content-writer, include target audience, key messages, and tone guidelines.
```

- [ ] **Step 3: Create parent MCP config**

Create `examples/deploy-gtm-agent/mcp.json`:

```json
{
  "mcpServers": {}
}
```

- [ ] **Step 4: Create parent skill**

Create `examples/deploy-gtm-agent/skills/competitor-analysis/SKILL.md`:

```markdown
---
name: competitor-analysis
description: >-
  Analyze competitors in a given market segment.
  Trigger on: competitive landscape, competitor analysis,
  market comparison, competitive positioning.
---

# Competitor Analysis

When asked to analyze competitors:

1. Identify the top 3-5 competitors in the target segment
2. For each competitor, assess:
   - Product positioning and key differentiators
   - Pricing model and tiers
   - Target audience and market share estimates
   - Strengths and weaknesses
3. Create a comparison matrix
4. Identify gaps and opportunities for differentiation
```

- [ ] **Step 5: Create market-researcher subagent**

Create `examples/deploy-gtm-agent/subagents/market-researcher/deepagents.toml`:

```toml
[agent]
name = "market-researcher"
description = "Researches market trends, competitors, and target audiences to inform GTM strategy"
model = "anthropic:claude-sonnet-4-6"
```

Create `examples/deploy-gtm-agent/subagents/market-researcher/AGENTS.md`:

```markdown
# Market Researcher

You are a market research specialist. Your job is to gather and synthesize market data to support go-to-market decisions.

## Focus Areas

- **Market sizing**: TAM, SAM, SOM estimates with methodology
- **Competitor analysis**: Product positioning, pricing, market share
- **Audience segmentation**: Demographics, psychographics, buying behavior
- **Trend analysis**: Industry trends, emerging technologies, regulatory changes

## Output Format

Structure your research as:

1. **Executive Summary** — Key findings in 2-3 sentences
2. **Methodology** — How you gathered and validated the data
3. **Findings** — Detailed analysis organized by topic
4. **Recommendations** — Actionable insights for the GTM strategy

## Guidelines

- Cite sources when possible
- Distinguish between hard data and estimates
- Flag areas of uncertainty or where more research is needed
- Keep analysis focused on what's actionable for go-to-market planning
```

- [ ] **Step 6: Create market-researcher skill**

Create `examples/deploy-gtm-agent/subagents/market-researcher/skills/analyze-market/SKILL.md`:

```markdown
---
name: analyze-market
description: >-
  Perform a market analysis for a product category or segment.
  Trigger on: market analysis, market size, TAM SAM SOM,
  market opportunity, industry analysis.
---

# Market Analysis

When asked to analyze a market:

1. Define the market boundaries (geography, segment, timeframe)
2. Estimate market size (TAM/SAM/SOM) with methodology
3. Identify key trends and growth drivers
4. Map the competitive landscape
5. Assess barriers to entry
6. Summarize the opportunity with a clear recommendation
```

- [ ] **Step 7: Commit**

```bash
git add examples/deploy-gtm-agent/
git commit -m "feat(examples): add GTM agent example with sync and async subagents"
```

---

### Task 9: Integration test — dry-run bundle with subagents

**Files:**
- Test: `libs/cli/tests/unit_tests/deploy/test_bundler.py`

- [ ] **Step 1: Write integration-style test**

Add to `TestBundle` in `test_bundler.py`:

```python
def test_bundle_with_subagents(self, tmp_path: Path) -> None:
    """Full bundle with sync and async subagents produces valid artifacts."""
    project = _minimal_project(tmp_path / "project")
    _add_subagent(
        project,
        "researcher",
        description="Research agent",
        skills={"search/SKILL.md": "# Search"},
    )
    _add_subagent(project, "coder", description="Coding agent")
    build = tmp_path / "build"
    config = DeployConfig(
        agent=AgentConfig(name="test-agent"),
        sandbox=SandboxConfig(),
        async_subagents=[
            AsyncSubAgentConfig(
                name="writer",
                description="Content writer",
                graph_id="writer-graph",
                url="https://example.com",
            ),
        ],
    )
    bundle(config, project, build)

    # Verify seed has subagents.
    seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))
    assert "subagents" in seed
    assert "researcher" in seed["subagents"]
    assert "coder" in seed["subagents"]
    assert seed["subagents"]["researcher"]["skills"]["/search/SKILL.md"] == "# Search"
    assert "async_subagents" in seed
    assert len(seed["async_subagents"]) == 1
    assert seed["async_subagents"][0]["name"] == "writer"

    # Verify generated deploy_graph.py is valid Python.
    graph_py = (build / "deploy_graph.py").read_text(encoding="utf-8")
    compile(graph_py, "<deploy_graph_subagents>", "exec")
    assert "SubAgent" in graph_py
    assert "AsyncSubAgent" in graph_py
    assert "_build_sync_subagents" in graph_py
    assert "_build_async_subagents" in graph_py
```

- [ ] **Step 2: Run the test**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/test_bundler.py::TestBundle::test_bundle_with_subagents -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run: `cd libs/cli && python -m pytest tests/unit_tests/deploy/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/cli/tests/unit_tests/deploy/test_bundler.py
git commit -m "test(deploy): add integration test for bundling with subagents"
```
