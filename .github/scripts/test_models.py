"""Tests for the GitHub Actions model matrix helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_SCRIPT = REPO_ROOT / ".github" / "scripts" / "models.py"
EVALS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evals.yml"
HARBOR_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "harbor.yml"


def _load_models_script() -> ModuleType:
    """Load `.github/scripts/models.py` as a module.

    The script lives outside any importable package, so import-by-path is the
    only way to exercise its internals from a test.
    """
    spec = importlib.util.spec_from_file_location("gha_models", MODELS_SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {MODELS_SCRIPT}"
        raise AssertionError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def models() -> ModuleType:
    """Module-scoped handle to the loaded `models.py` script."""
    return _load_models_script()


def test_eval_matrix_outputs_are_partitioned_by_provider(models: ModuleType) -> None:
    """Eval matrix outputs should queue each provider independently."""
    outputs = models._matrix_outputs(
        "eval",
        [
            "anthropic:claude-sonnet-4-6",
            "openrouter:moonshotai/kimi-k2.6",
            "new_provider:model-1",
        ],
    )

    assert outputs["anthropic_has_models"] is True
    assert outputs["openrouter_has_models"] is True
    assert outputs["other_has_models"] is True
    assert outputs["openai_has_models"] is False
    assert outputs["anthropic_matrix"] == {
        "include": [
            {
                "model": "anthropic:claude-sonnet-4-6",
                "provider": "anthropic",
                "artifact_key": "anthropic-claude-sonnet-4-6",
            }
        ]
    }
    assert outputs["openrouter_matrix"] == {
        "include": [
            {
                "model": "openrouter:moonshotai/kimi-k2.6",
                "provider": "openrouter",
                "artifact_key": "openrouter-moonshotai-kimi-k2.6",
            }
        ]
    }
    assert outputs["other_matrix"] == {
        "include": [
            {
                "model": "new_provider:model-1",
                "provider": "new_provider",
                "artifact_key": "new_provider-model-1",
            }
        ]
    }


def test_harbor_matrix_output_stays_flat(models: ModuleType) -> None:
    """Harbor should keep the existing single-matrix output contract."""
    outputs = models._matrix_outputs("harbor", ["openai:gpt-5.4"])

    assert outputs == {
        "matrix": {
            "include": [
                {
                    "model": "openai:gpt-5.4",
                    "provider": "openai",
                    "artifact_key": "openai-gpt-5.4",
                }
            ]
        }
    }


def test_eval_matrix_outputs_with_no_models(models: ModuleType) -> None:
    """Empty model list emits empty includes for every declared provider.

    The per-provider job `if:` guards in `evals.yml` are the only thing
    keeping GHA from rejecting a `matrix.include == []` configuration, so
    this lock-in test ensures the empty shape is preserved verbatim.
    """
    outputs = models._matrix_outputs("eval", [])

    assert outputs["matrix"] == {"include": []}
    for provider in models._EVAL_PROVIDER_OUTPUTS:
        assert outputs[f"{provider}_has_models"] is False
        assert outputs[f"{provider}_matrix"] == {"include": []}


def test_eval_outputs_cover_every_declared_provider(models: ModuleType) -> None:
    """Every name in `_EVAL_PROVIDER_OUTPUTS` must produce both output keys."""
    for provider in models._EVAL_PROVIDER_OUTPUTS:
        spec = f"{provider}:dummy" if provider != "other" else "unknown:dummy"
        outputs = models._matrix_outputs("eval", [spec])
        assert outputs[f"{provider}_has_models"] is True, provider
        assert outputs[f"{provider}_matrix"]["include"], provider


def test_eval_workflow_outputs_match_provider_constant(models: ModuleType) -> None:
    """`evals.yml` prep outputs must stay in sync with `_EVAL_PROVIDER_OUTPUTS`.

    Mirrors `test_release_options.py`: parses the workflow YAML and compares
    declared output names against the source set, so a drift in either
    direction (new provider, deleted provider, typo) fails fast.
    """
    workflow = yaml.safe_load(EVALS_WORKFLOW.read_text())
    declared = set(workflow["jobs"]["prep"]["outputs"].keys()) - {"matrix"}

    expected = {f"{p}_matrix" for p in models._EVAL_PROVIDER_OUTPUTS} | {
        f"{p}_has_models" for p in models._EVAL_PROVIDER_OUTPUTS
    }

    assert declared == expected, (
        "evals.yml prep outputs are out of sync with _EVAL_PROVIDER_OUTPUTS — "
        f"missing: {expected - declared}, extra: {declared - expected}"
    )


def test_eval_workflow_per_provider_jobs_match_provider_constant(
    models: ModuleType,
) -> None:
    """Each provider in `_EVAL_PROVIDER_OUTPUTS` has a matching `eval-*` job.

    The job name uses dashes (e.g. `eval-google-genai`) while the constant
    uses underscores (`google_genai`); compare with that mapping in mind.
    """
    workflow = yaml.safe_load(EVALS_WORKFLOW.read_text())
    job_names = set(workflow["jobs"].keys())

    for provider in models._EVAL_PROVIDER_OUTPUTS:
        job = f"eval-{provider.replace('_', '-')}"
        assert job in job_names, (
            f"_EVAL_PROVIDER_OUTPUTS includes {provider!r} but evals.yml is "
            f"missing job {job!r}"
        )


def test_has_models_serializes_to_lowercase_bool(models: ModuleType) -> None:
    """`_has_models` must serialize to `true`/`false` for GHA string compare.

    `evals.yml` gates each per-provider job on `... == 'true'`; if the JSON
    encoding of the python `bool` ever drifts (e.g., to Python `True`),
    every gate would silently evaluate false.
    """
    outputs = models._matrix_outputs("eval", ["anthropic:claude-sonnet-4-6"])
    assert json.dumps(outputs["anthropic_has_models"]) == "true"
    assert json.dumps(outputs["openai_has_models"]) == "false"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openrouter:moonshotai/kimi-k2.6", "openrouter-moonshotai-kimi-k2.6"),
        ("openrouter:foo//bar", "openrouter-foo-bar"),
        (":leading-colon", "leading-colon"),
        ("trailing-slash/", "trailing-slash"),
        ("anthropic:claude-opus-4-7", "anthropic-claude-opus-4-7"),
    ],
)
def test_artifact_key_handles_disallowed_characters(
    models: ModuleType, spec: str, expected: str
) -> None:
    """`_artifact_key` strips/collapses every char outside `[a-zA-Z0-9._-]`."""
    assert models._artifact_key(spec) == expected


def test_resolve_models_dedupes_repeated_specs(models: ModuleType) -> None:
    """`_resolve_models` deduplicates so `artifact_key` cannot collide downstream.

    Without this, a typo'd `models_override` like `openai:gpt-5.5,openai:gpt-5.5`
    would produce two matrix rows that race to upload artifacts under the same
    name and fail mid-run.
    """
    resolved = models._resolve_models(
        "eval", "anthropic:claude-sonnet-4-6,anthropic:claude-sonnet-4-6"
    )
    assert resolved == ["anthropic:claude-sonnet-4-6"]


def test_resolve_models_preserves_first_occurrence_order(
    models: ModuleType,
) -> None:
    """Dedupe keeps each spec at its first position — guards against `set()`.

    A future "simplification" to `list(set(specs))` would silently scramble
    the matrix order; this test pins the `dict.fromkeys` contract.
    """
    resolved = models._resolve_models(
        "eval",
        "anthropic:claude-sonnet-4-6,openai:gpt-5.5,anthropic:claude-sonnet-4-6,"
        "openai:gpt-5.5,google_genai:gemini-3.1-pro",
    )
    assert resolved == [
        "anthropic:claude-sonnet-4-6",
        "openai:gpt-5.5",
        "google_genai:gemini-3.1-pro",
    ]


def test_resolve_models_dedupes_preset_branch(models: ModuleType) -> None:
    """Dedupe applies to preset resolution too, not just manual `models_override`.

    The `_artifact_key` docstring promises uniqueness is enforced by
    `_resolve_models`; this test pins that promise across both code paths so
    a future REGISTRY edit that accidentally duplicates a spec won't blow up
    the matrix mid-run.
    """
    resolved = models._resolve_models("eval", "all")
    assert len(resolved) == len(set(resolved))


def test_matrix_outputs_rejects_colliding_artifact_keys(
    models: ModuleType,
) -> None:
    """Defense-in-depth: if dedupe is ever bypassed, `_matrix_outputs` raises.

    Bypasses `_resolve_models` to feed two distinct specs that slugify to the
    same key (`foo:a/b` and `foo:a-b` both become `foo-a-b`) — the actual
    failure mode the tripwire defends against. Asserts both the slug and the
    raw model specs appear in the message so a CI failure is self-diagnosing.
    """
    with pytest.raises(ValueError) as excinfo:
        models._matrix_outputs("eval", ["foo:a/b", "foo:a-b"])
    msg = str(excinfo.value)
    assert "Duplicate artifact_key" in msg
    assert "foo-a-b" in msg
    assert "foo:a/b" in msg
    assert "foo:a-b" in msg


def test_matrix_outputs_rejects_three_way_collision(
    models: ModuleType,
) -> None:
    """Three-way collision lists the offending key once, with all three specs.

    Exercises the `len(specs) > 1` branch in the collision detector — the
    list-of-models grouping must not split a 3+ collision into separate
    entries or duplicate the slug in the message.
    """
    with pytest.raises(ValueError) as excinfo:
        models._matrix_outputs(
            "eval",
            ["foo:a/b", "foo:a-b", "foo:a:b"],
        )
    msg = str(excinfo.value)
    # Slug appears exactly once; all three offending specs are named.
    assert msg.count("'foo-a-b'") == 1
    assert "foo:a/b" in msg
    assert "foo:a-b" in msg
    assert "foo:a:b" in msg


def test_provider_returns_whole_string_when_no_colon(models: ModuleType) -> None:
    """`_provider` falls through cleanly when the spec lacks a `:` separator.

    Upstream `_resolve_models` rejects colon-less specs, but a future caller
    of `_provider`/`_matrix_entry` might not — this lock-in test pins the
    behavior so a silent rerouting to `other` is at least visible in tests.
    """
    assert models._provider("anthropic:claude-foo") == "anthropic"
    assert models._provider("standalone-name") == "standalone-name"


def test_main_writes_per_provider_outputs_to_github_output(
    models: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` writes one line per output key with compact JSON values.

    GitHub Actions parses `key=value\\n` lines from `$GITHUB_OUTPUT`. Multi-line
    values would require heredoc syntax; this test guards against a future
    refactor to `json.dumps(..., indent=2)` and confirms `_has_models` is
    written as the lowercase string `true`/`false` that the workflow gates
    compare against.
    """
    output_file = tmp_path / "github_output"
    output_file.touch()

    monkeypatch.setenv("EVAL_MODELS", "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr("sys.argv", ["models.py", "eval"])

    models.main()

    written = output_file.read_text().splitlines()
    keyed = dict(line.split("=", 1) for line in written)

    assert keyed["anthropic_has_models"] == "true"
    assert keyed["openai_has_models"] == "false"
    assert keyed["other_has_models"] == "false"

    matrix = json.loads(keyed["matrix"])
    assert matrix["include"][0]["model"] == "anthropic:claude-sonnet-4-6"

    anthropic_matrix = json.loads(keyed["anthropic_matrix"])
    assert anthropic_matrix["include"][0]["provider"] == "anthropic"

    for line in written:
        assert "\n" not in line


def test_every_registered_model_has_display_labels(models: ModuleType) -> None:
    """Every `Model` in `REGISTRY` must declare non-empty display fields.

    These labels feed radar legends and `MODEL_GROUPS.md` provider headings;
    a blank entry would silently render an empty legend item or `() (N models)`.
    """
    for entry in models.REGISTRY:
        assert entry.display_name, f"empty display_name for {entry.spec!r}"
        assert entry.provider_label, f"empty provider_label for {entry.spec!r}"


def test_provider_label_is_uniform_within_a_provider(models: ModuleType) -> None:
    """All models sharing a `provider:` prefix must share one `provider_label`.

    The doc generator reads the label from the *first* match, so a mismatch
    would silently hide some models' label preference.
    """
    by_prefix: dict[str, set[str]] = {}
    for entry in models.REGISTRY:
        prefix = entry.spec.split(":", 1)[0]
        by_prefix.setdefault(prefix, set()).add(entry.provider_label)
    inconsistent = {p: ls for p, ls in by_prefix.items() if len(ls) > 1}
    assert not inconsistent, f"provider_label drift: {inconsistent}"


def test_display_name_helper_returns_curated_label(models: ModuleType) -> None:
    """`display_name` returns the curated label for a registered spec."""
    assert models.display_name("anthropic:claude-sonnet-4-6") == "Claude Sonnet 4.6"
    assert models.display_name("xai:grok-4") == "Grok 4"


def test_display_name_helper_falls_back_to_bare_model(models: ModuleType) -> None:
    """`display_name` falls back to the model portion when spec is unknown."""
    assert models.display_name("madeup:my-cool-model") == "my-cool-model"
    assert models.display_name("just-a-name") == "just-a-name"


def test_provider_label_helper_returns_curated_label(models: ModuleType) -> None:
    """`provider_label` returns the curated label for a registered spec."""
    assert models.provider_label("google_genai:gemini-3.1-pro-preview") == "Google"
    assert models.provider_label("xai:grok-4") == "xAI"


def test_provider_label_helper_falls_back_to_prefix(models: ModuleType) -> None:
    """`provider_label` falls back to the raw prefix for unknown specs."""
    assert models.provider_label("madeup_provider:foo") == "madeup_provider"


def _expected_dropdown_options(models: ModuleType) -> set[str]:
    """Return the full allowed `models:` dropdown set: REGISTRY & presets & providers.

    Mirrors the workflow's logic — a dropdown choice resolves to either an
    explicit spec, a preset name handled by `_resolve_models`, a provider
    prefix (also a preset), or the empty/`all` sentinels.
    """
    registry = {m.spec for m in models.REGISTRY}
    presets = {p for _, ps in models._PRESET_SECTIONS for p, _ in ps}  # noqa: SLF001
    providers = {m.spec.split(":", 1)[0] for m in models.REGISTRY}
    return registry | presets | providers | {"", "all"}


@pytest.mark.parametrize(
    "workflow_path",
    [EVALS_WORKFLOW, HARBOR_WORKFLOW],
    ids=lambda p: p.name,
)
def test_workflow_models_dropdown_matches_registry(
    models: ModuleType, workflow_path: Path
) -> None:
    """`models:` dropdown options must match REGISTRY & presets & providers.

    Catches two drift modes: (1) an orphan option that no longer resolves to a
    real spec/preset (silent fallthrough to `_resolve_models`'s empty-result
    error at workflow_dispatch time), and (2) a registered spec that wasn't
    surfaced in the dropdown so users can't pick it without typing into
    `models_override`.
    """
    workflow = yaml.safe_load(workflow_path.read_text())
    # PyYAML 1.1 coerces the bare YAML key `on:` to the boolean `True`.
    # Either form may appear depending on yaml lib version, so check both.
    triggers = workflow.get(True, workflow.get("on"))
    options = triggers["workflow_dispatch"]["inputs"]["models"]["options"]
    declared = {str(o) for o in options}
    expected = _expected_dropdown_options(models)

    orphan = declared - expected
    missing = expected - declared - {""}  # empty sentinel handled by default

    assert not orphan, (
        f"{workflow_path.name}: dropdown contains options not in REGISTRY/presets/providers: "
        f"{sorted(orphan)}"
    )
    assert not missing, (
        f"{workflow_path.name}: REGISTRY/presets/providers missing from dropdown: "
        f"{sorted(missing)}"
    )
