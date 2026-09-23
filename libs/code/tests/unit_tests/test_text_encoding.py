"""UTF-8 prompts, skills, and hooks must survive legacy Windows text encodings."""

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest

from deepagents_code import agent, config, model_config
from deepagents_code.hooks import legacy as legacy_hooks
from deepagents_code.skills import commands

_PACKAGE = Path(config.__file__).parent


def _set_text_encoding(monkeypatch: pytest.MonkeyPatch, code_page: str) -> None:
    """Emulate the default encoding while retaining real filesystem I/O."""

    def text_encoding(encoding: str | None, _stacklevel: int = 2) -> str:
        return code_page if encoding in (None, "locale") else encoding

    monkeypatch.setattr(io, "text_encoding", text_encoding)


@pytest.fixture(
    autouse=True,
    params=[
        "cp936",
        "cp932",
        "cp949",
        "cp950",
        "cp874",
        "cp1252",
        "cp1251",
        "cp1250",
        "cp1253",
    ],
)
def legacy_encoding(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply a legacy code page to otherwise unspecified Path text encodings."""
    _set_text_encoding(monkeypatch, request.param)


def _load_script(name: str) -> ModuleType:
    path = _PACKAGE / "built_in_skills" / "skill-creator" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_prompt_preserves_packaged_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy defaults must neither reject nor corrupt the real system prompt."""
    with monkeypatch.context() as utf8:
        _set_text_encoding(utf8, "utf-8")
        expected = agent.get_system_prompt("probe", has_tavily=False)

    assert agent.get_system_prompt("probe", has_tavily=False) == expected


def test_default_prompt_preserves_unicode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default prompt loader must also support non-ASCII template content."""
    content = "# Project notes\nUse → to describe changes — café 日本語.\n"
    (tmp_path / "default_agent_prompt.md").write_text(content, encoding="utf-8")
    monkeypatch.setattr(config, "__file__", str(tmp_path / "config.py"))

    assert config.get_default_coding_instructions() == content


def test_reset_agent_preserves_unicode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Copying an agent's instructions must preserve UTF-8 bytes."""
    source = tmp_path / "source"
    source.mkdir()
    content = "# Instructions\nUse → to describe changes — café 日本語.\n"
    (source / "AGENTS.md").write_text(content, encoding="utf-8")
    monkeypatch.setattr(agent, "user_deepagents_dir", lambda: tmp_path)

    agent.reset_agent("target", source_agent="source")

    assert (tmp_path / "target" / "AGENTS.md").read_text(encoding="utf-8") == content


def test_created_skill_round_trips_as_utf8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI-generated skills must be readable by UTF-8 skill consumers."""
    monkeypatch.setattr(commands, "ensure_user_skills_dir", lambda _: tmp_path)

    commands._create("example", "probe")

    path = tmp_path / "example" / "SKILL.md"
    assert path.read_text(encoding="utf-8") == commands._generate_template("example")


def test_skill_creator_preserves_unicode(tmp_path: Path) -> None:
    """The built-in creator must write UTF-8 templates and example resources."""
    creator = _load_script("init_skill")
    name = "café"

    path = creator.init_skill(name, tmp_path)

    assert path == tmp_path / name
    expected = creator.SKILL_TEMPLATE.format(skill_name=name, skill_title="Café")
    assert (path / "SKILL.md").read_text(encoding="utf-8") == expected
    assert name in (path / "scripts" / "example.py").read_text(encoding="utf-8")
    assert "Café" in (path / "references" / "api_reference.md").read_text(
        encoding="utf-8"
    )
    assert (path / "assets" / "example_asset.txt").read_text(
        encoding="utf-8"
    ) == creator.EXAMPLE_ASSET


def test_legacy_hooks_config_reads_utf8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hook commands with non-ASCII arguments must load exactly as written."""
    hooks = [{"command": ["notify-send", "完了 — café"]}]
    # `ensure_ascii=False` keeps the raw UTF-8 bytes a hand-edited file would
    # contain; `\u` escapes would make the file ASCII and hide the decoding.
    (tmp_path / "hooks.json").write_text(
        json.dumps({"hooks": hooks}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", tmp_path)

    assert legacy_hooks._load_hooks() == hooks


def test_skill_validator_reads_utf8(tmp_path: Path) -> None:
    """The standalone validator must accept UTF-8 frontmatter and skill text."""
    content = "---\nname: café\ndescription: 日本語 — café →\n---\n# Café\n"
    (tmp_path / "SKILL.md").write_text(content, encoding="utf-8")
    validator = _load_script("quick_validate")

    valid, message = validator.validate_skill(tmp_path)

    assert valid, message
