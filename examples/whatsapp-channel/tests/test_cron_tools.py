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
    def test_returns_three_tools(self, tools) -> None:
        assert set(tools.keys()) == {"create_job", "list_jobs", "remove_job"}


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
        # Prompt is omitted from list output
        assert "prompt" not in rows[0]


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
