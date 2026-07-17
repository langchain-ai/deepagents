"""Tests for the release dependency freshness helper."""

from __future__ import annotations

import json
from email.message import Message
from typing import TYPE_CHECKING, Self
from urllib.error import HTTPError, URLError

import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from urllib.request import Request
from check_dep_freshness import (
    BYPASS_LABEL,
    COMMENT_MARKER,
    PRERELEASE_POLICY_ENV,
    DependencyDeclaration,
    PrereleasePolicy,
    PyPIRequestError,
    StaleDependency,
    check_dependency_freshness,
    extract_minimum,
    fetch_pypi_json,
    freshness_markdown,
    includes_prereleases,
    latest_pypi_version,
    load_declarations,
    local_dependency_names,
    main,
    within_upper_bound,
)
from packaging.requirements import Requirement
from packaging.version import Version


class FakeResponse:
    """Minimal context-managed HTTP response for urllib client tests."""

    def __init__(self, payload: object, *, raw: bool = False) -> None:
        """Store a JSON payload or pre-encoded response body."""
        self.body = payload if raw else json.dumps(payload).encode()

    def __enter__(self) -> Self:
        """Return this fake response."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Leave the fake response context."""

    def read(self) -> bytes:
        """Return the configured response body."""
        assert isinstance(self.body, bytes)
        return self.body


class SequenceOpener:
    """Return or raise configured outcomes in order."""

    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        """Initialize response outcomes."""
        self.outcomes = outcomes
        self.requests: list[tuple[Request, float]] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        """Record a request and consume the next outcome."""
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def pypi_payload(**releases: list[dict[str, object]]) -> dict[str, object]:
    """Build a minimal PyPI JSON payload."""
    return {"releases": releases}


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("demo>=1.2,<2", Version("1.2")),
        ("demo~=1.4.5", Version("1.4.5")),
        ("demo==2.0.1", Version("2.0.1")),
        ("demo>=1,>=2,<3", Version("2")),
        ("demo==1.2.*,>=1", Version("1")),
        ("demo==1.2.*", None),
        ("demo<3", None),
        ("demo", None),
    ],
)
def test_extract_minimum(requirement: str, expected: Version | None) -> None:
    assert extract_minimum(Requirement(requirement).specifier) == expected


def test_prerelease_policy_defaults_to_bound_type() -> None:
    stable = Version("1.0")
    prerelease = Version("1.0a1")

    assert not includes_prereleases(PrereleasePolicy.BOUND, stable)
    assert includes_prereleases(PrereleasePolicy.BOUND, prerelease)
    assert includes_prereleases(PrereleasePolicy.ALWAYS, stable)
    assert not includes_prereleases(PrereleasePolicy.NEVER, prerelease)


def test_latest_pypi_version_ignores_prereleases_for_stable_bound() -> None:
    payload = pypi_payload(
        **{
            "1.0": [{"yanked": False}],
            "1.1rc1": [{"yanked": False}],
            "2.0.dev1": [{"yanked": False}],
        }
    )

    assert latest_pypi_version(payload, include_prereleases=False) == Version("1.0")
    assert latest_pypi_version(payload, include_prereleases=True) == Version("2.0.dev1")


def test_latest_pypi_version_ignores_invalid_empty_and_fully_yanked_releases() -> None:
    payload = pypi_payload(
        **{
            "not-a-version": [{"yanked": False}],
            "1.0": [],
            "1.1": [{"yanked": True}],
            "1.2": [{"yanked": True}, {"yanked": False}],
        }
    )

    assert latest_pypi_version(payload, include_prereleases=False) == Version("1.2")


def test_latest_pypi_version_rejects_missing_releases() -> None:
    with pytest.raises(TypeError, match="releases mapping"):
        latest_pypi_version({}, include_prereleases=False)


@pytest.mark.parametrize(
    ("requirement", "version", "expected"),
    [
        ("demo>=1", "9", True),
        ("demo>=1,<3", "2.9", True),
        ("demo>=1,<3", "3", False),
        ("demo~=1.4", "1.9", True),
        ("demo~=1.4", "2.0", False),
        ("demo==1.4", "2.0", False),
    ],
)
def test_within_upper_bound(requirement: str, version: str, expected: bool) -> None:
    assert within_upper_bound(Requirement(requirement), Version(version)) is expected


def test_local_dependency_names_only_includes_exclusively_local_sources() -> None:
    manifest = {
        "tool": {
            "uv": {
                "sources": {
                    "DeepAgents": {"path": "../deepagents", "editable": True},
                    "workspace-package": {"workspace": True},
                    "conditional-local": [
                        {"path": "../one", "marker": "sys_platform == 'darwin'"},
                        {"path": "../two", "marker": "sys_platform != 'darwin'"},
                    ],
                    "mixed": [
                        {"path": "../local"},
                        {"git": "https://example.test/repo"},
                    ],
                    "remote": {"git": "https://example.test/repo"},
                }
            }
        }
    }

    assert local_dependency_names(manifest) == frozenset(
        {"deepagents", "workspace-package", "conditional-local"}
    )


def test_load_declarations_reads_required_and_optional_and_skips_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = "libs/example/pyproject.toml"
    path = tmp_path / manifest_path
    path.parent.mkdir(parents=True)
    path.write_text(
        """
[project]
name = "example"
dependencies = [
    "deepagents>=1",
    "requests>=2,<3",
    "bare-package",
    "remote @ https://example.test/remote.whl",
]

[project.optional-dependencies]
extra = ["requests>=2,<3", "httpx~=0.28"]

[tool.uv.sources]
deepagents = { path = "../deepagents", editable = true }
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("check_dep_freshness.REPO_ROOT", tmp_path)

    declarations = load_declarations(manifest_path, "example")

    assert [(item.requirement.name, item.minimum) for item in declarations] == [
        ("requests", Version("2")),
        ("httpx", Version("0.28")),
    ]


def test_fetch_pypi_json_builds_canonical_request() -> None:
    opener = SequenceOpener([FakeResponse(pypi_payload(**{"1.0": [{}]}))])

    payload = fetch_pypi_json("Demo_Package", opener=opener, sleep=lambda _delay: None)

    assert latest_pypi_version(payload, include_prereleases=False) == Version("1.0")
    request, timeout = opener.requests[0]
    assert request.full_url == "https://pypi.org/pypi/demo-package/json"
    assert request.get_header("Accept") == "application/json"
    assert "deepagents" in request.get_header("User-agent")
    assert timeout == 5.0


def test_fetch_pypi_json_retries_transient_http_error() -> None:
    headers = Message()
    headers["Retry-After"] = "999"
    error = HTTPError("https://pypi.org", 503, "unavailable", headers, None)
    opener = SequenceOpener(
        [error, FakeResponse(pypi_payload(**{"1.0": [{"yanked": False}]}))]
    )
    delays: list[float] = []

    payload = fetch_pypi_json("demo", opener=opener, sleep=delays.append)

    assert latest_pypi_version(payload, include_prereleases=False) == Version("1.0")
    assert len(opener.requests) == 2
    assert delays == [5.0]


def test_fetch_pypi_json_retries_malformed_response() -> None:
    opener = SequenceOpener(
        [
            FakeResponse(b"not json", raw=True),
            FakeResponse(pypi_payload(**{"1.0": [{}]})),
        ]
    )

    payload = fetch_pypi_json("demo", opener=opener, sleep=lambda _delay: None)

    assert latest_pypi_version(payload, include_prereleases=False) == Version("1.0")
    assert len(opener.requests) == 2


def test_fetch_pypi_json_marks_exhausted_network_error_transient() -> None:
    opener = SequenceOpener([URLError("offline"), URLError("offline")])

    with pytest.raises(PyPIRequestError, match="offline") as exc_info:
        fetch_pypi_json(
            "demo",
            attempts=2,
            opener=opener,
            sleep=lambda _delay: None,
        )

    assert exc_info.value.transient
    assert len(opener.requests) == 2


def test_fetch_pypi_json_does_not_retry_not_found() -> None:
    error = HTTPError("https://pypi.org", 404, "not found", Message(), None)
    opener = SequenceOpener([error])

    with pytest.raises(PyPIRequestError, match="HTTP 404") as exc_info:
        fetch_pypi_json("demo", opener=opener, sleep=lambda _delay: None)

    assert not exc_info.value.transient
    assert len(opener.requests) == 1


def finding(*, within: bool = True) -> StaleDependency:
    """Build one dependency freshness finding."""
    return StaleDependency(
        manifest_path="libs/deepagents/pyproject.toml",
        package_name="deepagents",
        dependency_name="langchain",
        minimum=Version("1.0"),
        latest=Version("1.1"),
        within_upper_bound=within,
    )


def test_freshness_markdown_formats_table_policy_and_marker() -> None:
    markdown = freshness_markdown(
        [finding(within=False)],
        policy=PrereleasePolicy.BOUND,
        unavailable=("langsmith",),
        include_marker=True,
    )

    assert markdown.startswith(COMMENT_MARKER)
    assert "| `deepagents` | `langchain` | `1.0` | `1.1` | **No** |" in markdown
    assert "only when the declared minimum is itself a pre-release" in markdown
    assert "`langsmith`" in markdown
    assert BYPASS_LABEL in markdown


def _write_manifest(tmp_path: Path, manifest_path: str, dependency: str) -> None:
    path = tmp_path / manifest_path
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""
[project]
name = "example"
version = "0.1.0"
dependencies = ["{dependency}"]
""".strip(),
        encoding="utf-8",
    )


def _configure_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest_path: str,
) -> Path:
    output = tmp_path / "github_output"
    summary = tmp_path / "github_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr("check_dep_freshness.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_dep_freshness.load_release_packages",
        lambda: {"libs/example": "example"},
    )
    monkeypatch.setattr(
        "check_dep_freshness.changed_manifests",
        lambda _base, _head, _packages: [manifest_path],
    )
    return output


def test_check_dependency_freshness_emits_advisory_stale_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = "libs/example/pyproject.toml"
    _write_manifest(tmp_path, manifest_path, "demo>=1,<3")
    output = _configure_check(monkeypatch, tmp_path, manifest_path)
    calls: list[str] = []

    def fetcher(name: str) -> dict[str, object]:
        calls.append(name)
        return pypi_payload(**{"2.0": [{"yanked": False}]})

    assert check_dependency_freshness("base", "head", fetcher=fetcher) == 0

    written = output.read_text(encoding="utf-8")
    assert "stale<<" in written
    assert "\ntrue\n" in written
    assert "indeterminate<<" in written
    assert COMMENT_MARKER in written
    assert calls == ["demo"]
    assert (
        (tmp_path / "github_summary")
        .read_text(encoding="utf-8")
        .startswith("## Dependency minimums trail PyPI")
    )


def test_check_dependency_freshness_fetches_each_canonical_project_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = "libs/example/pyproject.toml"
    path = tmp_path / manifest_path
    path.parent.mkdir(parents=True)
    path.write_text(
        """
[project]
name = "example"
dependencies = ["demo>=1", "Demo>=0.9,<3"]

[project.optional-dependencies]
extra = ["demo~=1.0"]
""".strip(),
        encoding="utf-8",
    )
    _configure_check(monkeypatch, tmp_path, manifest_path)
    calls: list[str] = []

    def fetcher(name: str) -> dict[str, object]:
        calls.append(name)
        return pypi_payload(**{"2.0": [{"yanked": False}]})

    assert check_dependency_freshness("base", "head", fetcher=fetcher) == 0
    assert calls == ["demo"]


def test_check_dependency_freshness_warns_without_failing_on_transient_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = "libs/example/pyproject.toml"
    _write_manifest(tmp_path, manifest_path, "demo>=1")
    output = _configure_check(monkeypatch, tmp_path, manifest_path)

    def fetcher(_name: str) -> dict[str, object]:
        msg = "PyPI is unavailable"
        raise PyPIRequestError(msg, transient=True)

    assert check_dependency_freshness("base", "head", fetcher=fetcher) == 0

    written = output.read_text(encoding="utf-8")
    assert "stale<<" in written
    assert "indeterminate<<" in written
    assert written.count("\ntrue\n") == 1
    assert COMMENT_MARKER not in written
    assert "transient PyPI query failure" in capsys.readouterr().out


def test_check_dependency_freshness_noops_without_changed_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        "check_dep_freshness.load_release_packages",
        lambda: {"libs/example": "example"},
    )
    monkeypatch.setattr(
        "check_dep_freshness.changed_manifests",
        lambda _base, _head, _packages: [],
    )

    assert check_dependency_freshness("base", "head") == 0
    written = output.read_text(encoding="utf-8")
    assert written.count("\nfalse\n") == 2
    assert COMMENT_MARKER not in written


def test_main_rejects_invalid_prerelease_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BASE_SHA", "base")
    monkeypatch.setenv("HEAD_SHA", "head")
    monkeypatch.setenv(PRERELEASE_POLICY_ENV, "sometimes")

    assert main() == 2


def test_main_requires_pull_request_shas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)

    assert main() == 2


def test_declaration_dataclass_keeps_parsed_requirement() -> None:
    declaration = DependencyDeclaration(
        manifest_path="libs/example/pyproject.toml",
        package_name="example",
        requirement=Requirement("demo>=1"),
        minimum=Version("1"),
    )

    assert declaration.requirement.name == "demo"
