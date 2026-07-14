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
