"""Tests for cron.tools — agent-facing LangChain tools."""

from __future__ import annotations

import pytest

from cron.tools import build_cron_tools, origin_ctx


@pytest.fixture
def tools(jobs_path):
    return {t.name: t for t in build_cron_tools(jobs_path)}


def _invoke(tool, **kwargs):
    """Invoke a LangChain tool with a plain-dict input (version-agnostic)."""
    return tool.invoke(kwargs)


class TestBuildCronTools:
    def test_returns_expected_tools(self, tools) -> None:
        assert set(tools.keys()) == {
            "create_job", "list_jobs", "edit_job", "remove_job",
        }


class TestCreateJobTool:
    def test_creates_with_origin_from_context(self, tools, jobs_path) -> None:
        origin_ctx.set({"chat_id": "111", "message_id": "m1"})
        result = _invoke(tools["create_job"], prompt="test", schedule="30m")
        assert "id" in result
        assert result["schedule_display"] == "once in 30m"
        assert result["next_run_at"] is not None

    def test_error_when_context_unset(self, tools) -> None:
        # Reset via a fresh sentinel value
        origin_ctx.set({})
        result = _invoke(tools["create_job"], prompt="t", schedule="30m")
        assert "error" in result

    def test_surfaces_schedule_parse_error(self, tools) -> None:
        origin_ctx.set({"chat_id": "111", "message_id": None})
        result = _invoke(tools["create_job"], prompt="t", schedule="garbage")
        assert "error" in result


class TestListJobsTool:
    def test_lists_only_caller_chat(self, tools, jobs_path) -> None:
        from cron.jobs import create_job as db_create_job

        db_create_job(jobs_path, prompt="a", schedule="30m",
                      origin={"chat_id": "111", "message_id": None})
        db_create_job(jobs_path, prompt="b", schedule="30m",
                      origin={"chat_id": "222", "message_id": None})

        origin_ctx.set({"chat_id": "111", "message_id": None})
        rows = _invoke(tools["list_jobs"])
        assert len(rows) == 1
        assert rows[0]["name"] == "a"

    def test_returns_full_payload(self, tools, jobs_path) -> None:
        """Every stored field must be present so the agent can reason about
        a job and pick precise arguments for ``edit_job``."""
        from cron.jobs import create_job as db_create_job

        db_create_job(jobs_path, prompt="full prompt", schedule="every 15m",
                      origin={"chat_id": "111", "message_id": "m1"})
        origin_ctx.set({"chat_id": "111", "message_id": None})

        rows = _invoke(tools["list_jobs"])
        job = rows[0]
        assert job["prompt"] == "full prompt"
        assert job["enabled"] is True
        assert job["schedule"]["kind"] == "interval"
        assert job["repeat"] == {"times": None, "completed": 0}
        assert job["origin"] == {"chat_id": "111", "message_id": "m1"}

    def test_error_when_context_unset(self, tools) -> None:
        origin_ctx.set({})
        result = _invoke(tools["list_jobs"])
        assert isinstance(result, dict) and "error" in result


class TestEditJobTool:
    def _make_job(self, jobs_path, **overrides):
        from cron.jobs import create_job as db_create_job
        defaults = {
            "prompt": "initial",
            "schedule": "every 30m",
            "origin": {"chat_id": "111", "message_id": None},
        }
        defaults.update(overrides)
        return db_create_job(jobs_path, **defaults)

    def test_requires_chat_context(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path)
        origin_ctx.set({})
        result = _invoke(tools["edit_job"], job_id=j["id"], enabled=False)
        assert "error" in result

    def test_requires_at_least_one_field(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path)
        origin_ctx.set({"chat_id": "111", "message_id": None})
        result = _invoke(tools["edit_job"], job_id=j["id"])
        assert result["error"] == "Pass at least one field to edit."

    def test_partial_update_leaves_other_fields(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path, prompt="keep", schedule="every 10m")
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], enabled=False)
        assert result["enabled"] is False
        assert result["prompt"] == "keep"
        assert result["schedule"]["display"] == "every 10m"

    def test_reschedule_resets_next_run_at(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path, schedule="every 30m")
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], schedule="every 5m")
        assert result["schedule"]["kind"] == "interval"
        assert result["schedule"]["minutes"] == 5
        assert result["next_run_at"] != j["next_run_at"]

    def test_repeat_zero_clears_cap(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path, schedule="every 30m", repeat=3)
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], repeat=0)
        assert result["repeat"]["times"] is None

    def test_strips_double_quoted_schedule(self, tools, jobs_path) -> None:
        """Tool-call serialization sometimes double-wraps string args, so the
        model sends ``'"every 5m"'``. The tool must tolerate one quote layer."""
        j = self._make_job(jobs_path, schedule="every 30m")
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], schedule='"every 5m"')
        assert result["schedule"]["minutes"] == 5

    def test_strips_single_quoted_schedule(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path, schedule="every 30m")
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], schedule="'every 5m'")
        assert result["schedule"]["minutes"] == 5

    def test_strips_quotes_from_job_id(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path)
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=f'"{j["id"]}"', enabled=False)
        assert result["id"] == j["id"]
        assert result["enabled"] is False

    def test_strips_quotes_from_name_and_prompt(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path)
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(
            tools["edit_job"],
            job_id=j["id"],
            name='"HN digest"',
            prompt="'summarize top 3 HN posts'",
        )
        assert result["name"] == "HN digest"
        assert result["prompt"] == "summarize top 3 HN posts"

    def test_bad_schedule_returns_error(self, tools, jobs_path) -> None:
        j = self._make_job(jobs_path)
        origin_ctx.set({"chat_id": "111", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], schedule="garbage")
        assert "error" in result

    def test_unknown_id_returns_error(self, tools) -> None:
        origin_ctx.set({"chat_id": "111", "message_id": None})
        result = _invoke(tools["edit_job"], job_id="nope", enabled=False)
        assert result == {"error": "not found"}

    def test_refuses_cross_chat_edit(self, tools, jobs_path) -> None:
        from cron.jobs import load_jobs

        j = self._make_job(jobs_path, prompt="a",
                           origin={"chat_id": "111", "message_id": None})
        origin_ctx.set({"chat_id": "222", "message_id": None})

        result = _invoke(tools["edit_job"], job_id=j["id"], prompt="hacked")
        assert result == {"error": "not found"}
        assert load_jobs(jobs_path)[0]["prompt"] == "a"


class TestRemoveJobTool:
    def test_removes_own_chat_job(self, tools, jobs_path) -> None:
        from cron.jobs import create_job as db_create_job, load_jobs

        j = db_create_job(jobs_path, prompt="a", schedule="30m",
                          origin={"chat_id": "111", "message_id": None})
        origin_ctx.set({"chat_id": "111", "message_id": None})
        result = _invoke(tools["remove_job"], job_id=j["id"])
        assert result == {"removed": True, "id": j["id"]}
        assert load_jobs(jobs_path) == []

    def test_refuses_cross_chat_removal(self, tools, jobs_path) -> None:
        from cron.jobs import create_job as db_create_job, load_jobs

        j = db_create_job(jobs_path, prompt="a", schedule="30m",
                          origin={"chat_id": "111", "message_id": None})
        origin_ctx.set({"chat_id": "222", "message_id": None})
        result = _invoke(tools["remove_job"], job_id=j["id"])
        assert result == {"removed": False, "reason": "not found"}
        assert len(load_jobs(jobs_path)) == 1
