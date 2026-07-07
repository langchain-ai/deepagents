"""Tests for the NVIDIA Nemotron 3 Ultra harness profile."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.profiles.harness._nvidia_nemotron_3_ultra import (
    _DEFAULT_READ_LIMIT,
    _EMPTY_TOOL_PLACEHOLDER,
    _HARNESS_PROFILE_SUFFIX_MARKER,
    ChatNVIDIAMessageCompatibilityMiddleware,
    EntityResolutionGuardMiddleware,
    FinalAnswerGuardMiddleware,
    FollowupDisciplineMiddleware,
    ModelRateLimitRetryMiddleware,
    NemotronPolicyNudgeMiddleware,
    NemotronProgressBudgetMiddleware,
    NemotronReasoningTagCleanupMiddleware,
    NemotronTextToolCallParser,
    NemotronToolCallShim,
    ReadFileContinuationNoticeMiddleware,
    _tool_name_is_domain,
    _tool_name_is_mutation,
    register,
)
from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_NEMOTRON_ULTRA_MODEL_SPECS: tuple[str, ...] = (
    "NVIDIA:nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
    "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-bf16",
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b",
    "nebius:nvidia/Nemotron-3-Ultra-550b-a55b",
    "together:nvidia/nemotron-3-ultra-550b-a55b",
)


def _runtime(tool_call_id: str = "call_1") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _request(name: str, args: dict[str, object]) -> ToolCallRequest:
    return ToolCallRequest(
        runtime=_runtime(),
        tool_call={"id": "call_1", "name": name, "args": args},
        state={},
        tool=None,
    )


def _numbered_lines(count: int) -> str:
    return "\n".join(f"{index}\tline {index}" for index in range(count))


def test_tool_call_shim_repairs_file_path_args_and_empty_results() -> None:
    """Nemotron's common `path` arg should be remapped before tool execution."""
    middleware = NemotronToolCallShim()
    captured: list[ToolCallRequest] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        captured.append(request)
        return ToolMessage(content="", tool_call_id="call_1")

    result = middleware.wrap_tool_call(_request("read_file", {"path": "/big.txt"}), handler)

    assert isinstance(result, ToolMessage)
    assert result.content == _EMPTY_TOOL_PLACEHOLDER
    assert captured[0].tool_call["args"] == {"file_path": "/big.txt", "limit": _DEFAULT_READ_LIMIT}


def test_tool_call_shim_does_not_delete_existing_files(tmp_path: Path) -> None:
    """The shim should not delete real files to recover from write errors."""
    middleware = NemotronToolCallShim()
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:  # noqa: ARG001
        nonlocal calls
        calls += 1
        if target.exists():
            return ToolMessage(
                content=f"Error: cannot write {target} because it already exists",
                tool_call_id="call_1",
            )
        msg = "handler should not be retried by deleting the existing file"
        raise AssertionError(msg)

    result = middleware.wrap_tool_call(
        _request("write_file", {"path": str(target), "content": "new"}),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert "already exists" in result.content
    assert target.read_text(encoding="utf-8") == "old"
    assert calls == 1


def test_read_file_continuation_notice_marks_exact_limit_results() -> None:
    """Exactly-at-limit `read_file` output should tell the model to keep paging."""
    middleware = ReadFileContinuationNoticeMiddleware()

    def handler(request: ToolCallRequest) -> ToolMessage:  # noqa: ARG001
        return ToolMessage(content="1\talpha\n2\tbeta\n3\tgamma", tool_call_id="call_1")

    result = middleware.wrap_tool_call(
        _request("read_file", {"file_path": "/x.txt", "limit": 3, "offset": 9}),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert "read_file returned 3 lines starting at offset 9" in result.content
    assert "offset=12" in result.content


def test_read_file_continuation_notice_uses_shim_default_limit() -> None:
    """The continuation notice should describe the same limit sent to `read_file`."""
    shim = NemotronToolCallShim()
    notice = ReadFileContinuationNoticeMiddleware()

    def handler(request: ToolCallRequest) -> ToolMessage:
        assert request.tool_call["args"]["limit"] == _DEFAULT_READ_LIMIT
        return ToolMessage(
            content=_numbered_lines(_DEFAULT_READ_LIMIT),
            tool_call_id="call_1",
        )

    result = shim.wrap_tool_call(
        _request("read_file", {"file_path": "/big.txt"}),
        lambda request: notice.wrap_tool_call(request, handler),
    )

    assert isinstance(result, ToolMessage)
    assert f"read_file returned {_DEFAULT_READ_LIMIT} lines starting at offset 0" in result.content
    assert f"offset={_DEFAULT_READ_LIMIT}" in result.content


class FakeRateLimitError(Exception):
    """Fake provider 429 for model-call retry tests."""

    status_code: int = 429


def test_model_rate_limit_retry_retries_transient_429() -> None:
    """Model 429s should be retried without swallowing non-rate-limit errors."""
    middleware = ModelRateLimitRetryMiddleware(retry_delays=(0.0,))
    calls = 0

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = "rate limit exceeded"
            raise FakeRateLimitError(msg)
        return AIMessage(content="ok")

    result = middleware.wrap_model_call(object(), handler)

    assert isinstance(result, AIMessage)
    assert result.content == "ok"
    assert calls == 2


def test_progress_budget_stops_repeated_tool_call_loop() -> None:
    """Repeated identical tool calls should short-circuit instead of looping."""
    middleware = NemotronProgressBudgetMiddleware()
    repeated = {
        "name": "get_current_incident_id",
        "args": {},
        "id": "call_1",
        "type": "tool_call",
    }
    messages = [
        HumanMessage("What is the current incident?"),
        AIMessage(content="", tool_calls=[{**repeated, "id": "call_1"}]),
        ToolMessage(content="41017", tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{**repeated, "id": "call_2"}]),
        ToolMessage(content="41017", tool_call_id="call_2"),
        AIMessage(content="", tool_calls=[{**repeated, "id": "call_3"}]),
        ToolMessage(content="41017", tool_call_id="call_3"),
    ]
    request = SimpleNamespace(state={"messages": messages}, messages=messages)

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        msg = "handler should not be called after the progress budget is exceeded"
        raise AssertionError(msg)

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert result.name == "nemotron_progress_budget"
    assert "get_current_incident_id" in result.content


def test_progress_budget_allows_nonconsecutive_repeated_reads() -> None:
    """Repeated reads separated by edits should not look like an infinite loop."""
    middleware = NemotronProgressBudgetMiddleware(max_repeated_tool_calls=3)
    read_call_1 = {"name": "read_file", "args": {"file_path": "/x.txt"}, "id": "call_1", "type": "tool_call"}
    edit_call = {
        "name": "edit_file",
        "args": {"file_path": "/x.txt", "old_string": "old", "new_string": "new"},
        "id": "call_2",
        "type": "tool_call",
    }
    read_call_2 = {"name": "read_file", "args": {"file_path": "/x.txt"}, "id": "call_3", "type": "tool_call"}
    read_call_3 = {"name": "read_file", "args": {"file_path": "/x.txt"}, "id": "call_4", "type": "tool_call"}
    messages = [
        HumanMessage("Iteratively edit and reread the file."),
        AIMessage(content="", tool_calls=[read_call_1]),
        ToolMessage(content="old", tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[edit_call]),
        ToolMessage(content="updated", tool_call_id="call_2"),
        AIMessage(content="", tool_calls=[read_call_2]),
        ToolMessage(content="new", tool_call_id="call_3"),
        AIMessage(content="", tool_calls=[read_call_3]),
        ToolMessage(content="new", tool_call_id="call_4"),
    ]
    request = SimpleNamespace(state={"messages": messages}, messages=messages)

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        return AIMessage(content="continue")

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert result.content == "continue"


def test_progress_budget_accepts_profile_specific_limits() -> None:
    """Step budgets should be configurable instead of fixed module behavior."""
    middleware = NemotronProgressBudgetMiddleware(max_model_calls=2)
    messages = [
        HumanMessage("Find the deployment status."),
        AIMessage(content="", tool_calls=[{"name": "status_probe", "args": {}, "id": "call_1", "type": "tool_call"}]),
        ToolMessage(content="queued", tool_call_id="call_1"),
        AIMessage(content="still checking"),
    ]
    request = SimpleNamespace(state={"messages": messages}, messages=messages)

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        msg = "handler should not be called after the configured model budget is exceeded"
        raise AssertionError(msg)

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert result.name == "nemotron_progress_budget"
    assert result.response_metadata["nemotron_progress_budget_reason"] == "2 model turns"


def test_progress_budget_counts_active_turn_only() -> None:
    """Old checkpoint history should not consume the current turn's step budget."""
    middleware = NemotronProgressBudgetMiddleware(max_model_calls=2)
    old_messages: list[object] = [HumanMessage("old task")]
    old_messages.extend(AIMessage(content=f"old response {index}") for index in range(20))
    current_messages = [HumanMessage("new task"), AIMessage(content="working")]
    messages = [*old_messages, *current_messages]
    request = SimpleNamespace(state={"messages": messages}, messages=messages)

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        return AIMessage(content="continued")

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert result.content == "continued"


def test_domain_tool_preference_triggers_for_non_file_domain_request() -> None:
    """Initial non-file questions should prefer task tools over filesystem tools."""
    messages = [HumanMessage("Which service has the most firing alerts?")]
    request = SimpleNamespace(
        state={"messages": messages},
        messages=messages,
        tools=[
            {"name": "ls"},
            {"name": "glob"},
            {"name": "alerts_catalog"},
            {"name": "metric_probe"},
        ],
    )

    assert NemotronPolicyNudgeMiddleware._should_prefer_domain_tools(request)


def test_filesystem_request_nudge_triggers_for_named_file_request() -> None:
    """Initial file-content requests should not end as no-access answers."""

    class FakeRequest(SimpleNamespace):
        def override(self, **kwargs: object) -> FakeRequest:
            return FakeRequest(**{**self.__dict__, **kwargs})

    messages = [HumanMessage("Read the entirety of summarization.py, 500 lines at a time, and summarize it.")]
    request = FakeRequest(
        state={"messages": messages},
        messages=messages,
        tools=[{"name": "ls"}, {"name": "glob"}, {"name": "read_file"}],
    )

    captured: list[object] = []

    def handler(request: object) -> AIMessage:
        captured.append(request)
        return AIMessage(content="ok")

    result = NemotronPolicyNudgeMiddleware().wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert captured
    assert "Do not answer that you lack access" in captured[0].messages[-1].content


def test_mutation_detection_uses_action_tokens_not_vendor_names() -> None:
    """Mutation detection should be based on action verbs, not brand/tool families."""
    assert _tool_name_is_mutation("postChannelMessage")
    assert _tool_name_is_mutation("create_issue")
    assert _tool_name_is_mutation("revoke_access")
    assert _tool_name_is_mutation("charge_card")
    assert not _tool_name_is_domain("delete")
    assert not _tool_name_is_mutation("delete")
    assert not _tool_name_is_mutation("get_charge")
    assert not _tool_name_is_mutation("get_post")
    assert not _tool_name_is_mutation("get_transfer")
    assert not _tool_name_is_mutation("search_archive")
    assert not _tool_name_is_mutation("github_lookup_issue")
    assert not _tool_name_is_mutation("postal_code_lookup")
    assert not _tool_name_is_mutation("write_file")
    assert not _tool_name_is_mutation("write_todos")


def test_text_tool_call_parser_repairs_function_blocks() -> None:
    """Text-form function calls should become structured tool calls."""
    message = AIMessage(
        content=("Run this.\n<function=grep><parameter name=pattern>MAGIC</parameter><parameter name=path>/workspace/tmp</parameter></function>")
    )

    repaired = NemotronTextToolCallParser._repair_message(message)

    assert repaired.content == "Run this."
    assert repaired.tool_calls[0]["name"] == "grep"
    assert repaired.tool_calls[0]["args"] == {
        "pattern": "MAGIC",
        "path": "/workspace/tmp",
    }


def test_text_tool_call_parser_respects_available_function_tools() -> None:
    """Function-block repairs should not synthesize hidden or unavailable tools."""
    message = AIMessage(content="<function=execute><parameter name=command>pytest -q</parameter></function>")

    repaired = NemotronTextToolCallParser._repair_message(message, {"execute"})
    untouched = NemotronTextToolCallParser._repair_message(message, {"read_file"})

    assert repaired.content == ""
    assert repaired.tool_calls[0]["name"] == "execute"
    assert untouched.content == message.content
    assert untouched.tool_calls == []


def test_text_tool_call_parser_repairs_json_shell_alias() -> None:
    """JSON shell aliases should become `execute` tool calls."""
    message = AIMessage(content='{"tool": "bash", "cmd": "pytest -q"}')

    repaired = NemotronTextToolCallParser._repair_message(message)

    assert repaired.content == ""
    assert repaired.tool_calls[0]["name"] == "execute"
    assert repaired.tool_calls[0]["args"] == {"command": "pytest -q"}


def test_text_tool_call_parser_respects_available_json_tools() -> None:
    """JSON tool repairs should be gated to tools exposed on the request."""
    message = AIMessage(content='{"tool": "bash", "cmd": "pytest -q"}')

    repaired = NemotronTextToolCallParser._repair_message(message, {"execute"})
    untouched = NemotronTextToolCallParser._repair_message(message, {"read_file"})

    assert repaired.content == ""
    assert repaired.tool_calls[0]["name"] == "execute"
    assert untouched.content == message.content
    assert untouched.tool_calls == []


def test_chatnvidia_message_compatibility_mirrors_tool_call_metadata() -> None:
    """Standard LangChain tool calls should survive ChatNVIDIA serialization."""
    middleware = ChatNVIDIAMessageCompatibilityMiddleware()
    messages = [
        HumanMessage("Look this up."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup",
                    "args": {"id": 1},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ok", name="lookup", tool_call_id="call_1"),
    ]

    class FakeRequest(SimpleNamespace):
        def override(self, **kwargs: object) -> FakeRequest:
            return FakeRequest(**{**self.__dict__, **kwargs})

    request = FakeRequest(messages=messages, state={"messages": messages}, tools=[{"name": "lookup"}])
    captured: list[FakeRequest] = []

    def handler(request: FakeRequest) -> AIMessage:
        captured.append(request)
        return AIMessage(content="done")

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, AIMessage)
    assert captured
    repaired_ai = captured[0].messages[1]
    repaired_tool = captured[0].messages[2]
    assert repaired_ai.additional_kwargs["tool_calls"][0]["function"]["name"] == "lookup"
    assert repaired_tool.additional_kwargs["name"] == "lookup"


def test_reasoning_tag_cleanup_removes_tags_and_preserves_reasoning() -> None:
    """Reasoning tags should not remain in normal assistant content."""
    middleware = NemotronReasoningTagCleanupMiddleware()

    def handler(request: object) -> AIMessage:  # noqa: ARG001
        return AIMessage(content="<think>hidden reasoning</think>\nVisible answer")

    result = middleware.wrap_model_call(object(), handler)

    assert isinstance(result, AIMessage)
    assert result.content == "Visible answer"
    assert result.additional_kwargs["reasoning_content"] == "hidden reasoning"


def test_text_tool_call_parser_repairs_alternate_blocks_only_for_available_tools() -> None:
    """Alternate raw tool-call markup should be gated to request tools."""
    message = AIMessage(content=("<function>\n<name=get_service_name</name>\n<parameter>\n<service_id>:0\n</parameter>\n</function>\n</tool_call>"))

    repaired = NemotronTextToolCallParser._repair_message(message, {"get_service_name"})
    untouched = NemotronTextToolCallParser._repair_message(message, {"get_user_name"})

    assert repaired.content == ""
    assert repaired.tool_calls[0]["name"] == "get_service_name"
    assert repaired.tool_calls[0]["args"] == {"service_id": "0"}
    assert untouched.content == message.content
    assert untouched.tool_calls == []


def test_final_answer_guard_preserves_mutation_literals() -> None:
    """Final answers after mutation tools should include exact key literals."""
    middleware = FinalAnswerGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("Notify the #deployments channel that v2.0 has been released"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "slack_post_channel",
                        "args": {
                            "channel": "#deployments",
                            "message": "v2.0 has been released!",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Posted to #deployments", tool_call_id="call_1"),
            AIMessage(content="Done - posted to #deployments channel."),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["nemotron_final_guard_fired"] is True
    assert "v2.0" in update["messages"][0].content


def test_final_answer_guard_preserves_mutation_titles() -> None:
    """Mutation titles should be treated as final-answer literals."""
    middleware = FinalAnswerGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("Create issue titled Fix memory leak and notify incidents"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "github_create_issue",
                        "args": {
                            "repo": "org/backend",
                            "title": "Fix memory leak",
                            "body": "OOM in prod",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Created issue 'Fix memory leak'", tool_call_id="call_1"),
            AIMessage(content="Done. Created the issue and notified #incidents."),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert "Fix memory leak" in update["messages"][0].content


def test_final_answer_guard_respects_exact_final_text_requests() -> None:
    """Exact final-answer requests should not be rewritten after tool success."""
    middleware = FinalAnswerGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("Notify #deployments that v2.0 shipped, then reply with the single word DONE."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "slack_post_channel",
                        "args": {
                            "channel": "#deployments",
                            "message": "v2.0 shipped",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Posted to #deployments", tool_call_id="call_1"),
            AIMessage(content="DONE"),
        ]
    }

    assert middleware.after_agent(state, None) is None


def test_final_answer_guard_leaves_informative_mutation_answer_alone() -> None:
    """Informative mutation answers should not be rewritten to quote full results."""
    middleware = FinalAnswerGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("Email the weekly status report to manager@company.com with subject 'Week 10 Status'"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "gmail_send_email",
                        "args": {
                            "to": "manager@company.com",
                            "subject": "Week 10 Status",
                            "body": "Weekly Status Report - Week 10",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Sent email to manager@company.com: Week 10 Status - Weekly Status Report - Week 10",
                tool_call_id="call_1",
            ),
            AIMessage(content='Sent the weekly status report to manager@company.com with subject "Week 10 Status".'),
        ]
    }

    assert middleware.after_agent(state, None) is None


def test_followup_guard_rewrites_redundant_schedule_questions() -> None:
    """Recurring requests should only be rewritten after redundant questions."""
    middleware = FollowupDisciplineMiddleware()
    state = {
        "messages": [
            HumanMessage("Send a report to my team every week"),
            AIMessage(content="What day should I send it, what time should I send it, and who should receive it?"),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["nemotron_followup_guard_fired"] is True
    assert "schedule, cadence" in update["messages"][0].content


def test_followup_guard_rewrites_analysis_without_goal_question() -> None:
    """Vague analysis follow-ups should ask for goal as well as source."""
    middleware = FollowupDisciplineMiddleware()
    state = {
        "messages": [
            HumanMessage("Can you analyze my data?"),
            AIMessage(content="What file, database, API, or pasted data should I use?"),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert "analysis goal" in update["messages"][0].content


def test_followup_guard_allows_legitimate_source_question() -> None:
    """Source/scope questions should survive when the user did not give a source."""
    middleware = FollowupDisciplineMiddleware()
    state = {
        "messages": [
            HumanMessage("Can you prepare a weekly report?"),
            AIMessage(content="Which source should I use for the report?"),
        ]
    }

    assert middleware.after_agent(state, None) is None


def test_followup_guard_rewrites_scope_question_when_scope_supplied() -> None:
    """Scope questions should be rewritten only when the user supplied scope."""
    middleware = FollowupDisciplineMiddleware()
    state = {
        "messages": [
            HumanMessage("Prepare a weekly report for the current project."),
            AIMessage(content="Which project should I use?"),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert "source, or scope" in update["messages"][0].content


def test_followup_guard_respects_exact_final_text_requests() -> None:
    """Follow-up rewrites should not override a satisfied exact answer request."""
    middleware = FollowupDisciplineMiddleware()
    state = {
        "messages": [
            HumanMessage("Send a report to my team every week. Reply with the single word DONE."),
            AIMessage(content="DONE"),
        ]
    }

    assert middleware.after_agent(state, None) is None


def test_action_commit_nudge_fires_after_user_approval_with_prior_tool_context() -> None:
    """Approved actions should get a human-message nudge before more explanation."""
    middleware = NemotronPolicyNudgeMiddleware()
    state = {
        "messages": [
            HumanMessage("Can you check reservation ABC123?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_reservation_details",
                        "args": {"reservation_id": "ABC123"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content='{"reservation_id":"ABC123"}', tool_call_id="call_1"),
            HumanMessage("Go ahead and cancel it now."),
        ]
    }

    update = middleware.before_model(state, None)

    assert update is not None
    assert update["nemotron_action_nudged"]
    assert "perform an action now" in update["messages"][0].content


def test_tool_chain_nudge_fires_after_information_lookup_before_action() -> None:
    """Chained lookup/action requests should not repeat the lookup indefinitely."""
    middleware = NemotronPolicyNudgeMiddleware()
    state = {
        "messages": [
            HumanMessage("Search for release notes and email a summary to team@co.com"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "release notes"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Top results", tool_call_id="call_1"),
        ]
    }

    update = middleware.before_model(state, None)

    assert update is not None
    assert update["nemotron_tool_chain_nudged"] is True
    assert "chained action" in update["messages"][0].content


def test_final_answer_guard_rewrites_vague_mutation_result() -> None:
    """Final answers after mutations should communicate the concrete tool result."""
    middleware = FinalAnswerGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("Cancel reservation ABC123."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "cancel_reservation",
                        "args": {"reservation_id": "ABC123"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content='{"reservation_id":"ABC123","status":"cancelled","refund":"$25"}', tool_call_id="call_1"),
            AIMessage(content="Done."),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert update["jump_to"] == "model"
    assert "cancel_reservation" in update["messages"][0].content
    assert "cancelled" in update["messages"][0].content


def test_domain_tool_nudge_fires_after_dead_end_filesystem_search() -> None:
    """Dead-end filesystem exploration should return to domain/API tools."""
    middleware = NemotronPolicyNudgeMiddleware()
    state = {
        "messages": [
            HumanMessage("Which service has the most firing alerts?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "incident_catalog",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="[41017, 41029]", tool_call_id="call_1"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "grep",
                        "args": {"pattern": "alert"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="No matches found", tool_call_id="call_2"),
        ]
    }

    update = middleware.before_model(state, None)

    assert update is not None
    assert update["nemotron_domain_tool_nudged"] is True
    assert "non-filesystem API/domain tools" in update["messages"][0].content


def test_conversation_transition_nudges_on_new_long_context_file_task() -> None:
    """Long follow-on file work should receive a compact-conversation reminder."""
    middleware = NemotronPolicyNudgeMiddleware()
    state = {
        "messages": [
            HumanMessage("Read and summarize /first.py"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/first.py"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="1\talpha", tool_call_id="call_1"),
            AIMessage(content="summary"),
            HumanMessage("Thanks. Move on to a new task: read /second.py and summarize it."),
            AIMessage(content="thinking"),
            HumanMessage("Actually do the same for another file /third.py."),
        ]
    }

    update = middleware.before_model(state, None)

    assert update is not None
    assert update["nemotron_transition_nudged"] is True
    assert "compact_conversation" in update["messages"][0].content


def test_entity_resolution_guard_keeps_current_entity_branch_bound() -> None:
    """Current entity branches should resolve the branch-specific display name."""
    middleware = EntityResolutionGuardMiddleware()
    state = {
        "messages": [
            HumanMessage("What service is affected by the current incident?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_incident_id",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="41017", tool_call_id="call_1"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_incident_service",
                        "args": {"incident_id": 41017},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="8514", tool_call_id="call_2"),
            AIMessage(content="The current incident affects checkout-web."),
        ]
    }

    update = middleware.after_agent(state, None)

    assert update is not None
    assert update["nemotron_entity_guard_fired"] is True
    assert update["jump_to"] == "model"
    assert "service_id 8514" in update["messages"][0].content


def test_register_adds_ultra3_profiles_for_supported_providers() -> None:
    """Every supported Nemotron Ultra spec should receive profile data."""
    original = dict(_HARNESS_PROFILES)
    try:
        _HARNESS_PROFILES.clear()
        register()

        for spec in _EXPECTED_NEMOTRON_ULTRA_MODEL_SPECS:
            profile = _HARNESS_PROFILES[spec]
            middleware = profile.materialize_extra_middleware()

            assert _HARNESS_PROFILE_SUFFIX_MARKER in (profile.system_prompt_suffix or "")
            assert "whole/full file" in profile.tool_description_overrides["read_file"]
            assert [entry.name for entry in middleware] == [
                "NemotronProgressBudgetMiddleware",
                "NemotronPolicyNudgeMiddleware",
                "NemotronToolCallShim",
                "ReadFileContinuationNoticeMiddleware",
                "ToolRetryMiddleware",
                "ModelRateLimitRetryMiddleware",
                "ChatNVIDIAMessageCompatibilityMiddleware",
                "NemotronReasoningTagCleanupMiddleware",
                "NemotronTextToolCallParser",
                "FollowupDisciplineMiddleware",
                "EntityResolutionGuardMiddleware",
                "FinalAnswerGuardMiddleware",
            ]
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)
