import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enumerate_tasks as et  # noqa: E402


def test_local_task_names_lists_dirs_with_task_toml(tmp_path):
    (tmp_path / "cb-cloud-1").mkdir()
    (tmp_path / "cb-cloud-1" / "task.toml").write_text("")
    (tmp_path / "cb-cloud-0").mkdir()
    (tmp_path / "cb-cloud-0" / "task.toml").write_text("")
    (tmp_path / "not-a-task").mkdir()  # no task.toml -> excluded
    (tmp_path / "dataset.toml").write_text("")  # file, not a task dir
    assert et.local_task_names(str(tmp_path)) == ["cb-cloud-0", "cb-cloud-1"]


def test_local_task_names_empty_raises(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="No local Harbor tasks"):
        et.local_task_names(str(tmp_path))


def test_tau3_subset_is_registered_synthetic():
    # "tau3-subset" is not an org/name registry package, so it must resolve via
    # a synthetic resolver rather than the registry client.
    assert "tau3-subset" in et.SYNTHETIC_DATASETS


def test_main_routes_synthetic_dataset_not_registry(tmp_path, monkeypatch):
    import pytest

    out = tmp_path / "names.txt"
    monkeypatch.setitem(et.SYNTHETIC_DATASETS, "tau3-subset", lambda: ["a", "b", "c"])
    monkeypatch.setenv("ENUM_DATASET", "tau3-subset")
    monkeypatch.setenv("ENUM_OUTPUT", str(out))
    monkeypatch.delenv("ENUM_DATASET_PATH", raising=False)
    monkeypatch.setattr(
        et,
        "registry_task_names",
        lambda _ref: pytest.fail("registry must not be queried for a synthetic dataset"),
    )
    assert et.main() == 0
    assert out.read_text().split() == ["a", "b", "c"]
