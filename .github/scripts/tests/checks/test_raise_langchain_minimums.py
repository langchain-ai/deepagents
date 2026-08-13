"""Tests for the scheduled LangChain minimum-version updater."""

from __future__ import annotations

from pathlib import Path

import pytest
import raise_langchain_minimums
import yaml
from check_release_deps import load_release_packages
from packaging.requirements import Requirement
from packaging.version import Version
from raise_langchain_minimums import (
    ManifestScope,
    _apply_replacements,
    _compatible_release_version,
    _in_scope,
    _latest_compatible_version,
    _load_scope,
    _plan_manifest,
    _project_requirement_strings,
    _raise_lower_bound,
    _raiseable_specifier,
    _select_manifests,
    edits_markdown,
    stale_lock_dirs,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _versions(*raw: str) -> list[Version]:
    return [Version(item) for item in raw]


def _raise(requirement_string: str, new_minimum: str) -> str:
    requirement = Requirement(requirement_string)
    specifier = _raiseable_specifier(requirement.specifier)
    assert specifier is not None
    return _raise_lower_bound(requirement_string, specifier, Version(new_minimum))


class TestLatestCompatibleVersion:
    def test_respects_upper_bound(self) -> None:
        """A newer major release cannot make an existing range unsatisfiable."""
        requirement = Requirement("langchain>=1.3.14,<2.0.0")
        versions = _versions("1.3.14", "1.9.0", "2.0.0")

        assert _latest_compatible_version(requirement, versions) == Version("1.9.0")

    def test_respects_compatible_release_range(self) -> None:
        """Compatible-release constraints remain in force when raising a floor."""
        requirement = Requirement("langchain~=1.3.14")
        versions = _versions("1.3.14", "1.3.20", "1.4.0")

        assert _latest_compatible_version(requirement, versions) == Version("1.3.20")

    def test_returns_none_when_nothing_in_range(self) -> None:
        requirement = Requirement("langchain>=1.0,<2.0")

        assert _latest_compatible_version(requirement, _versions("2.5.0")) is None


class TestRaiseableSpecifier:
    def test_picks_strongest_floor_not_iteration_order(self) -> None:
        """`SpecifierSet` iteration order is an implementation detail.

        The clause chosen must be the strongest floor, matching the bound
        `extract_minimum` reports, regardless of how `packaging` orders clauses.
        """
        specifier = _raiseable_specifier(Requirement("langchain~=1.2,>=1.4").specifier)

        assert specifier is not None
        assert (specifier.operator, specifier.version) == (">=", "1.4")

    @pytest.mark.parametrize("requirement_string", ["langchain==1.0", "langchain<2.0"])
    def test_returns_none_without_a_raiseable_floor(
        self, requirement_string: str
    ) -> None:
        assert _raiseable_specifier(Requirement(requirement_string).specifier) is None


class TestRaiseLowerBound:
    @pytest.mark.parametrize(
        ("requirement_string", "expected"),
        [
            ("langchain>=1.0", "langchain>=1.6.0"),
            ("langchain>=1.0,<2.0", "langchain>=1.6.0,<2.0"),
            # Whitespace around the operator is legal PEP 508; `packaging`
            # normalizes it away, so a verbatim match on the parsed clause fails.
            ("langchain >= 1.0", "langchain >=1.6.0"),
            ("langchain>= 1.0, <2.0", "langchain>=1.6.0, <2.0"),
            # Extras and markers survive untouched.
            (
                'langchain[extra]>=1.0 ; python_version>="3.11"',
                'langchain[extra]>=1.6.0 ; python_version>="3.11"',
            ),
        ],
    )
    def test_rewrites_floor_preserving_the_rest(
        self, requirement_string: str, expected: str
    ) -> None:
        assert _raise(requirement_string, "1.6.0") == expected

    def test_compatible_release_ceiling_is_not_narrowed(self) -> None:
        """`~=1.2` allows all of 1.x; raising it must not clamp to 1.9.x.

        `~=X.Y` means `>=X.Y, ==X.*`, so rewriting the version token in place to
        a three-component release would silently tighten the ceiling to
        `==1.9.*` while claiming upper bounds are preserved.
        """
        raised = _raise("langchain~=1.2", "1.9.4")

        assert raised == "langchain~=1.9"
        assert Requirement(raised).specifier.contains(Version("1.9.9"))

    def test_compatible_release_patch_series_is_kept(self) -> None:
        raised = _raise("langchain~=1.3.14", "1.3.20")

        assert raised == "langchain~=1.3.20"


class TestCompatibleReleaseVersion:
    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            ("1.2", "1.9.4", "1.9"),
            ("1.3.14", "1.3.20", "1.3.20"),
            ("1.3.14", "1.3.20.1", "1.3.20"),
        ],
    )
    def test_truncates_to_the_original_component_count(
        self, old: str, new: str, expected: str
    ) -> None:
        assert _compatible_release_version(old, Version(new)) == expected


class TestApplyReplacements:
    def test_prefix_requirement_does_not_corrupt_its_neighbour(self) -> None:
        """A requirement that is a prefix of another must not bleed into it.

        Unanchored substring replacement would rewrite the `langsmith>=0.10.9`
        inside `langsmith>=0.10.9,<0.11.0`, producing a spliced version that was
        never checked against PyPI — and which one wins depended on dict order.
        """
        text = 'a = ["langsmith>=0.10.9"]\nb = ["langsmith>=0.10.9,<0.11.0"]\n'
        replacements = {
            "langsmith>=0.10.9": "langsmith>=0.11.5",
            "langsmith>=0.10.9,<0.11.0": "langsmith>=0.10.12,<0.11.0",
        }

        result = _apply_replacements(text, replacements, "pyproject.toml")

        assert result == (
            'a = ["langsmith>=0.11.5"]\nb = ["langsmith>=0.10.12,<0.11.0"]\n'
        )

    def test_comment_mentions_are_left_alone(self) -> None:
        text = (
            '# Keep langchain>=1.0 in sync with the SDK.\ndeps = ["langchain>=1.0"]\n'
        )

        result = _apply_replacements(
            text, {"langchain>=1.0": "langchain>=1.6.0"}, "pyproject.toml"
        )

        assert result == (
            '# Keep langchain>=1.0 in sync with the SDK.\ndeps = ["langchain>=1.6.0"]\n'
        )

    def test_single_quoted_literals_are_rewritten(self) -> None:
        result = _apply_replacements(
            "deps = ['langchain>=1.0']",
            {"langchain>=1.0": "langchain>=1.6.0"},
            "pyproject.toml",
        )

        assert result == "deps = ['langchain>=1.6.0']"

    def test_unmatched_replacement_raises(self) -> None:
        """A reported edit that cannot be applied must fail, not be logged as done."""
        with pytest.raises(ValueError, match="no matching quoted literal"):
            _apply_replacements(
                'deps = ["langchain >= 1.0"]',
                {"langchain>=1.0": "langchain>=1.6.0"},
                "pyproject.toml",
            )


class TestPlanManifest:
    def _scope(self, tmp_path: Path, body: str) -> ManifestScope:
        """Write a manifest into a `REPO_ROOT`-patched tmp dir and parse it."""
        (tmp_path / "pyproject.toml").write_text(body, encoding="utf-8")
        return _load_scope("pyproject.toml")

    def test_floor_is_never_raised_past_an_upper_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`deepagents>=0.7.0,<0.8.0` must not become `>=0.8.0,<0.8.0`.

        Five partner manifests carry exactly that bound, so an unclamped raise
        would make them unsatisfiable the day `deepagents` 0.8.0 ships.
        """
        monkeypatch.setattr(raise_langchain_minimums, "REPO_ROOT", tmp_path)
        scope = self._scope(
            tmp_path,
            '[project]\nname = "x"\ndependencies = ["deepagents>=0.7.0,<0.8.0"]\n',
        )

        plan = _plan_manifest(
            scope, {"deepagents": _versions("0.7.0", "0.7.9", "0.8.0", "0.9.0")}
        )

        assert [edit.new_requirement for edit in plan.edits] == [
            "deepagents>=0.7.9,<0.8.0"
        ]
        assert 'dependencies = ["deepagents>=0.7.9,<0.8.0"]' in plan.new_text

    def test_own_name_and_local_sources_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(raise_langchain_minimums, "REPO_ROOT", tmp_path)
        scope = self._scope(
            tmp_path,
            "[project]\n"
            'name = "deepagents-code"\n'
            'dependencies = ["deepagents-code>=1.0", "deepagents>=1.0"]\n'
            "\n"
            "[tool.uv.sources]\n"
            'deepagents = { path = "../deepagents", editable = true }\n',
        )

        assert scope.requirements == ()

    def test_exact_pins_are_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(raise_langchain_minimums, "REPO_ROOT", tmp_path)
        scope = self._scope(
            tmp_path, '[project]\nname = "x"\ndependencies = ["deepagents==0.7.0"]\n'
        )

        plan = _plan_manifest(scope, {"deepagents": _versions("0.9.0")})

        assert plan.edits == ()
        assert plan.new_text == scope.text

    def test_dependency_groups_are_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(raise_langchain_minimums, "REPO_ROOT", tmp_path)
        scope = self._scope(
            tmp_path,
            '[project]\nname = "x"\ndependencies = []\n'
            "\n"
            "[dependency-groups]\n"
            'test = ["langsmith>=0.1.0"]\n',
        )

        plan = _plan_manifest(scope, {"langsmith": _versions("0.4.0")})

        assert [edit.new_requirement for edit in plan.edits] == ["langsmith>=0.4.0"]


class TestStaleLockDirs:
    def test_includes_path_dependents_transitively(self) -> None:
        """Editing `libs/deepagents` invalidates every lockfile that path-depends on it.

        `check_lockfiles.yml` only inspects packages a PR diff touches, so a
        dependent lockfile omitted here goes stale on `main`.
        """
        stale = stale_lock_dirs(["libs/deepagents/pyproject.toml"])

        assert "libs/deepagents" in stale
        assert "libs/evals" in stale
        assert "libs/code" in stale
        assert "libs/partners/quickjs" in stale

    def test_leaf_package_only_invalidates_itself(self) -> None:
        assert stale_lock_dirs(["libs/evals/pyproject.toml"]) == ["libs/evals"]


class TestSelectManifests:
    def test_resolves_a_release_label(self) -> None:
        packages = load_release_packages()
        label = packages["libs/deepagents"]

        assert _select_manifests(label, packages) == ["libs/deepagents/pyproject.toml"]

    def test_resolves_a_release_path(self) -> None:
        packages = load_release_packages()

        assert _select_manifests("libs/deepagents", packages) == [
            "libs/deepagents/pyproject.toml"
        ]

    def test_all_selects_every_release_package(self) -> None:
        packages = load_release_packages()

        assert _select_manifests("all", packages) == sorted(
            f"{path}/pyproject.toml" for path in packages
        )

    def test_unknown_package_returns_none(self) -> None:
        assert _select_manifests("nope", load_release_packages()) is None


class TestProjectRequirementStrings:
    def test_reads_dependencies_and_optional_dependencies(self) -> None:
        project = {
            "dependencies": ["langchain>=1.0", 42],
            "optional-dependencies": {"extra": ["langsmith>=0.1"]},
        }

        assert _project_requirement_strings(project) == [
            "langchain>=1.0",
            "langsmith>=0.1",
        ]


class TestInScope:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("langchain", True),
            ("langchain-openai", True),
            ("langgraph-cli", True),
            ("langsmith", True),
            ("deepagents-acp", True),
            ("pytest", False),
            # Bare prefix matching, with no hyphen boundary.
            ("langchainhub", True),
        ],
    )
    def test_prefix_matching(self, name: str, expected: bool) -> None:
        requirement = Requirement(f"{name}>=1.0")

        assert _in_scope(requirement, name, frozenset(), None) is expected

    def test_url_requirements_are_excluded(self) -> None:
        requirement = Requirement("langchain @ https://example.invalid/x.whl")

        assert not _in_scope(requirement, "langchain", frozenset(), None)


class TestEditsMarkdown:
    def test_renders_one_table_row_per_edit(self) -> None:
        edit = raise_langchain_minimums.RequirementEdit(
            manifest_path="libs/code/pyproject.toml",
            dependency_name="langsmith",
            old_requirement="langsmith>=0.1.0",
            new_requirement="langsmith>=0.4.0",
        )

        assert edits_markdown([edit], heading="Raised 1 minimum(s):") == (
            "Raised 1 minimum(s):\n"
            "\n"
            "| Manifest | Dependency | Change |\n"
            "|---|---|---|\n"
            "| `libs/code/pyproject.toml` | `langsmith` | "
            "`langsmith>=0.1.0` → `langsmith>=0.4.0` |"
        )


def test_workflow_options_match_release_labels() -> None:
    """The dispatch dropdown must stay in sync with the release package labels."""
    workflow = REPO_ROOT / ".github" / "workflows" / "raise_langchain_minimums.yml"
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    options = set(data[True]["workflow_dispatch"]["inputs"]["package"]["options"])
    labels = set(load_release_packages().values())

    assert options - {"all"} == labels
