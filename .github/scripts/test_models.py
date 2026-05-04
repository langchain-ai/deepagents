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
                "artifact_key": "000-anthropic-claude-sonnet-4-6",
            }
        ]
    }
    assert outputs["openrouter_matrix"] == {
        "include": [
            {
                "model": "openrouter:moonshotai/kimi-k2.6",
                "provider": "openrouter",
                "artifact_key": "001-openrouter-moonshotai-kimi-k2.6",
            }
        ]
    }
    assert outputs["other_matrix"] == {
        "include": [
            {
                "model": "new_provider:model-1",
                "provider": "new_provider",
                "artifact_key": "002-new_provider-model-1",
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
                    "artifact_key": "000-openai-gpt-5.4",
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
    ("index", "spec", "expected"),
    [
        (0, "openrouter:moonshotai/kimi-k2.6", "000-openrouter-moonshotai-kimi-k2.6"),
        (5, "openrouter:foo//bar", "005-openrouter-foo-bar"),
        (12, ":leading-colon", "012-leading-colon"),
        (99, "trailing-slash/", "099-trailing-slash"),
        (7, "anthropic:claude-opus-4-7", "007-anthropic-claude-opus-4-7"),
    ],
)
def test_artifact_key_handles_disallowed_characters(
    models: ModuleType, index: int, spec: str, expected: str
) -> None:
    """`_artifact_key` strips/collapses every char outside `[a-zA-Z0-9._-]`."""
    assert models._artifact_key(index, spec) == expected


def test_artifact_key_index_disambiguates_identical_slugs(
    models: ModuleType,
) -> None:
    """Same spec at different indexes still produces unique keys."""
    spec = "anthropic:claude-sonnet-4-6"
    assert models._artifact_key(0, spec) != models._artifact_key(1, spec)


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
