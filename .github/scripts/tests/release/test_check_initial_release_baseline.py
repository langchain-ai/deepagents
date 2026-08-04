"""Tests for check_initial_release_baseline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_initial_release_baseline
from check_initial_release_baseline import (
    BAD_INITIAL_BASELINE,
    RECOMMENDED_INITIAL_BASELINE,
    _load_json,
    _packages,
    main,
    new_bad_baseline_packages,
)

SCRIPT = Path(check_initial_release_baseline.__file__)

# An arbitrary already-released version for the pre-existing package. Any value
# other than 0.0.1 works; the detector only special-cases the 0.0.1 baseline.
EXISTING_VERSION = "1.0.0"

BASE_CONFIG = {
    "packages": {
        "libs/deepagents": {"component": "deepagents"},
    }
}
BASE_MANIFEST = {"libs/deepagents": EXISTING_VERSION}
HEAD_CONFIG = {
    "packages": {
        "libs/deepagents": {"component": "deepagents"},
        "libs/partners/vercel": {"component": "langchain-vercel-sandbox"},
    }
}
HEAD_MANIFEST = {
    "libs/deepagents": EXISTING_VERSION,
    "libs/partners/vercel": BAD_INITIAL_BASELINE,
}


def test_new_package_at_bad_baseline_is_flagged() -> None:
    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=HEAD_CONFIG,
        head_manifest=HEAD_MANIFEST,
    ) == ["langchain-vercel-sandbox"]


def test_existing_package_at_bad_baseline_is_not_flagged() -> None:
    base_config = {
        "packages": {
            **BASE_CONFIG["packages"],
            "libs/talon": {"component": "deepagents-talon"},
        }
    }
    base_manifest = {**BASE_MANIFEST, "libs/talon": BAD_INITIAL_BASELINE}
    head_config = {"packages": {**base_config["packages"]}}
    head_manifest = {**base_manifest}

    assert new_bad_baseline_packages(
        base_config=base_config,
        base_manifest=base_manifest,
        head_config=head_config,
        head_manifest=head_manifest,
    ) == []


def test_package_in_base_config_but_new_to_manifest_is_flagged() -> None:
    # The config entry already landed on base, but the manifest baseline is
    # only now being added at 0.0.1. The path is "new" by virtue of the
    # manifest alone, which exercises the `or` in the new-path filter: an `and`
    # would wrongly treat the path as already-known and skip it.
    base_config = {
        "packages": {
            **BASE_CONFIG["packages"],
            "libs/partners/vercel": {"component": "langchain-vercel-sandbox"},
        }
    }
    # vercel is absent from the manifest baseline despite being in base config.
    base_manifest = {**BASE_MANIFEST}
    head_config = {"packages": {**base_config["packages"]}}
    head_manifest = {**base_manifest, "libs/partners/vercel": BAD_INITIAL_BASELINE}

    assert new_bad_baseline_packages(
        base_config=base_config,
        base_manifest=base_manifest,
        head_config=head_config,
        head_manifest=head_manifest,
    ) == ["langchain-vercel-sandbox"]


def test_new_package_at_recommended_baseline_is_not_flagged() -> None:
    head_manifest = {
        **HEAD_MANIFEST,
        "libs/partners/vercel": RECOMMENDED_INITIAL_BASELINE,
    }

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=HEAD_CONFIG,
        head_manifest=head_manifest,
    ) == []


def test_new_package_at_later_baseline_is_not_flagged() -> None:
    head_manifest = {**HEAD_MANIFEST, "libs/partners/vercel": "0.0.2"}

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=HEAD_CONFIG,
        head_manifest=head_manifest,
    ) == []


def test_new_package_in_head_config_only_is_not_flagged() -> None:
    # Present in head config but absent from head manifest: there is no
    # manifest baseline to equal 0.0.1, so it cannot trigger the bug and is
    # correctly ignored.
    head_config = {
        "packages": {
            **HEAD_CONFIG["packages"],
            "libs/partners/cfgonly": {"component": "langchain-cfgonly"},
        }
    }

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=head_config,
        head_manifest=HEAD_MANIFEST,
    ) == ["langchain-vercel-sandbox"]


def test_new_package_in_head_manifest_only_is_flagged_by_path() -> None:
    # Present in head manifest at 0.0.1 but absent from head config: it is
    # still flagged, falling back to the raw path as the component name.
    head_manifest = {**HEAD_MANIFEST, "libs/partners/orphan": BAD_INITIAL_BASELINE}

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=HEAD_CONFIG,
        head_manifest=head_manifest,
    ) == ["langchain-vercel-sandbox", "libs/partners/orphan"]


def test_new_package_without_component_falls_back_to_path() -> None:
    head_config = {
        "packages": {
            **BASE_CONFIG["packages"],
            "libs/partners/vercel": {},
        }
    }

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=head_config,
        head_manifest=HEAD_MANIFEST,
    ) == ["libs/partners/vercel"]


def test_multiple_new_bad_baseline_packages_are_sorted() -> None:
    # `new_paths` is a set, so its iteration order is hash-seed dependent and
    # the reverse-alphabetical dict order below is discarded before sorting.
    # Using several offenders makes it vanishingly unlikely (~1/n!) that a
    # set-order result would coincidentally match the sorted expectation, so
    # dropping `sorted()` reliably fails this test. The workflow's list
    # rendering relies on the stable order.
    components = ["echo", "delta", "charlie", "bravo", "alpha"]
    head_config = {
        "packages": {
            **BASE_CONFIG["packages"],
            **{
                f"libs/partners/{name}": {"component": f"langchain-{name}"}
                for name in components
            },
        }
    }
    head_manifest = {
        **BASE_MANIFEST,
        **{f"libs/partners/{name}": BAD_INITIAL_BASELINE for name in components},
    }

    assert new_bad_baseline_packages(
        base_config=BASE_CONFIG,
        base_manifest=BASE_MANIFEST,
        head_config=head_config,
        head_manifest=head_manifest,
    ) == [
        "langchain-alpha",
        "langchain-bravo",
        "langchain-charlie",
        "langchain-delta",
        "langchain-echo",
    ]


def test_main_clean_when_no_offenders(capsys, tmp_path) -> None:
    # The most common real outcome: a correctly-authored PR using the
    # recommended baseline. main() must exit 0, print an empty array, and stay
    # silent on stderr.
    head_manifest = {
        **HEAD_MANIFEST,
        "libs/partners/vercel": RECOMMENDED_INITIAL_BASELINE,
    }
    base_config = tmp_path / "base-config.json"
    base_manifest = tmp_path / "base-manifest.json"
    head_config = tmp_path / "head-config.json"
    head_manifest_path = tmp_path / "head-manifest.json"
    base_config.write_text(json.dumps(BASE_CONFIG), encoding="utf-8")
    base_manifest.write_text(json.dumps(BASE_MANIFEST), encoding="utf-8")
    head_config.write_text(json.dumps(HEAD_CONFIG), encoding="utf-8")
    head_manifest_path.write_text(json.dumps(head_manifest), encoding="utf-8")

    rc = main(base_config, base_manifest, head_config, head_manifest_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == []
    assert captured.err == ""


def test_main_array_input_returns_2(capsys, tmp_path) -> None:
    # A syntactically valid JSON array where an object is expected exercises the
    # TypeError branch of main() (distinct from the malformed-JSON ValueError
    # branch covered below).
    base_config = tmp_path / "base-config.json"
    base_manifest = tmp_path / "base-manifest.json"
    head_config = tmp_path / "head-config.json"
    head_manifest = tmp_path / "head-manifest.json"
    base_config.write_text("[]", encoding="utf-8")
    base_manifest.write_text(json.dumps(BASE_MANIFEST), encoding="utf-8")
    head_config.write_text(json.dumps(HEAD_CONFIG), encoding="utf-8")
    head_manifest.write_text(json.dumps(HEAD_MANIFEST), encoding="utf-8")

    assert main(base_config, base_manifest, head_config, head_manifest) == 2
    assert "::error::" in capsys.readouterr().err


def test_load_json_rejects_array(tmp_path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError):
        _load_json(path)


def test_load_json_rejects_malformed(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read JSON object"):
        _load_json(path)


def test_load_json_rejects_unreadable(tmp_path) -> None:
    # A nonexistent path exercises the OSError half of the caught tuple
    # (distinct from the JSONDecodeError half above), mapping to the same
    # ValueError contract that main()'s exit-2 path depends on.
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(ValueError, match="Could not read JSON object"):
        _load_json(missing)


def test_packages_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="packages"):
        _packages({"packages": []})


def test_packages_defaults_to_empty_when_key_absent() -> None:
    assert _packages({}) == {}


def test_main_prints_offenders_as_json(capsys, tmp_path) -> None:
    base_config = tmp_path / "base-config.json"
    base_manifest = tmp_path / "base-manifest.json"
    head_config = tmp_path / "head-config.json"
    head_manifest = tmp_path / "head-manifest.json"
    base_config.write_text(json.dumps(BASE_CONFIG), encoding="utf-8")
    base_manifest.write_text(json.dumps(BASE_MANIFEST), encoding="utf-8")
    head_config.write_text(json.dumps(HEAD_CONFIG), encoding="utf-8")
    head_manifest.write_text(json.dumps(HEAD_MANIFEST), encoding="utf-8")

    rc = main(base_config, base_manifest, head_config, head_manifest)
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == ["langchain-vercel-sandbox"]
    assert BAD_INITIAL_BASELINE in captured.err


def test_main_invalid_json_returns_2(capsys, tmp_path) -> None:
    base_config = tmp_path / "base-config.json"
    base_manifest = tmp_path / "base-manifest.json"
    head_config = tmp_path / "head-config.json"
    head_manifest = tmp_path / "head-manifest.json"
    base_config.write_text("{not json", encoding="utf-8")
    base_manifest.write_text(json.dumps(BASE_MANIFEST), encoding="utf-8")
    head_config.write_text(json.dumps(HEAD_CONFIG), encoding="utf-8")
    head_manifest.write_text(json.dumps(HEAD_MANIFEST), encoding="utf-8")

    assert main(base_config, base_manifest, head_config, head_manifest) == 2
    assert "::error::" in capsys.readouterr().err


def test_cli_wrong_arg_count_exits_2() -> None:
    # The __main__ guard prints usage and exits 2 when not given exactly four
    # file arguments, before any analysis runs.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "only-one-arg"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_cli_with_four_valid_args_exits_0(tmp_path) -> None:
    # The happy-path __main__ dispatch into main() with four real files: a
    # clean run prints an empty JSON array and exits 0.
    paths = []
    for name, payload in (
        ("base-config", BASE_CONFIG),
        ("base-manifest", BASE_MANIFEST),
        ("head-config", BASE_CONFIG),
        ("head-manifest", BASE_MANIFEST),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *paths],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == []
