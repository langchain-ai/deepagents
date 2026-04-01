"""Unit tests for structured session compaction."""

from deepagents.middleware.structured_compaction import (
    COMPACT_CONTINUATION_PREAMBLE,
    COMPACT_DIRECT_RESUME_INSTRUCTION,
    COMPACT_RECENT_MESSAGES_NOTE,
    COMPACT_SUMMARY_PROMPT,
    format_compact_summary,
    get_compact_continuation_message,
    get_compact_prompt,
)


class TestGetCompactPrompt:
    """Tests for the compaction prompt builder."""

    def test_contains_nine_sections(self):
        prompt = get_compact_prompt()
        for section in [
            "1. Primary Request and Intent",
            "2. Key Technical Concepts",
            "3. Files and Code Sections",
            "4. Errors and fixes",
            "5. Problem Solving",
            "6. All user messages",
            "7. Pending Tasks",
            "8. Current Work",
            "9. Optional Next Step",
        ]:
            assert section in prompt

    def test_contains_analysis_instruction(self):
        prompt = get_compact_prompt()
        assert "<analysis>" in prompt
        assert "Chronologically analyze" in prompt

    def test_contains_example_structure(self):
        prompt = get_compact_prompt()
        assert "<example>" in prompt
        assert "</example>" in prompt
        assert "<summary>" in prompt

    def test_custom_instructions_appended(self):
        prompt = get_compact_prompt("Focus on database changes.")
        assert "Focus on database changes." in prompt
        assert "Additional Instructions:" in prompt

    def test_empty_custom_instructions_ignored(self):
        prompt = get_compact_prompt("")
        assert "Additional Instructions:" not in prompt

    def test_none_custom_instructions_ignored(self):
        prompt = get_compact_prompt(None)
        assert "Additional Instructions:" not in prompt


class TestFormatCompactSummary:
    """Tests for summary formatting."""

    def test_strips_analysis_block(self):
        raw = "<analysis>Thinking...</analysis>\n<summary>The result</summary>"
        result = format_compact_summary(raw)
        assert "Thinking..." not in result
        assert "The result" in result

    def test_replaces_summary_tags_with_header(self):
        raw = "<summary>Some content here</summary>"
        result = format_compact_summary(raw)
        assert result == "Summary:\nSome content here"

    def test_plain_text_passthrough(self):
        raw = "Just plain text without XML tags"
        result = format_compact_summary(raw)
        assert result == "Just plain text without XML tags"

    def test_collapses_multiple_blank_lines(self):
        raw = "<summary>line1\n\n\n\nline2</summary>"
        result = format_compact_summary(raw)
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_handles_empty_analysis(self):
        raw = "<analysis></analysis><summary>Content</summary>"
        result = format_compact_summary(raw)
        assert result == "Summary:\nContent"

    def test_multiline_analysis_stripped(self):
        raw = "<analysis>\nStep 1: think\nStep 2: more thinking\n</analysis>\n<summary>\n1. Primary Request:\n   User wanted X\n</summary>"
        result = format_compact_summary(raw)
        assert "think" not in result
        assert "Primary Request" in result


class TestGetCompactContinuationMessage:
    """Tests for continuation message generation."""

    def test_includes_preamble(self):
        msg = get_compact_continuation_message("Test summary")
        assert msg.startswith(COMPACT_CONTINUATION_PREAMBLE)

    def test_formats_summary(self):
        msg = get_compact_continuation_message("<summary>My summary content</summary>")
        assert "Summary:\nMy summary content" in msg

    def test_includes_resume_instruction_by_default(self):
        msg = get_compact_continuation_message("Test")
        assert COMPACT_DIRECT_RESUME_INSTRUCTION in msg

    def test_suppress_follow_up_false(self):
        msg = get_compact_continuation_message("Test", suppress_follow_up_questions=False)
        assert COMPACT_DIRECT_RESUME_INSTRUCTION not in msg

    def test_includes_transcript_path(self):
        msg = get_compact_continuation_message("Test", transcript_path="/history/session.md")
        assert "/history/session.md" in msg
        assert "read the full transcript" in msg

    def test_no_transcript_path(self):
        msg = get_compact_continuation_message("Test", transcript_path=None)
        assert "read the full transcript" not in msg

    def test_recent_messages_preserved(self):
        msg = get_compact_continuation_message("Test", recent_messages_preserved=True)
        assert COMPACT_RECENT_MESSAGES_NOTE in msg

    def test_recent_messages_not_preserved(self):
        msg = get_compact_continuation_message("Test", recent_messages_preserved=False)
        assert COMPACT_RECENT_MESSAGES_NOTE not in msg

    def test_full_message_structure(self):
        """Verify the complete structure matches the expected format."""
        msg = get_compact_continuation_message(
            "<summary>Full test</summary>",
            suppress_follow_up_questions=True,
            transcript_path="/conv/thread.md",
            recent_messages_preserved=True,
        )
        # Order: preamble → summary → transcript → recent note → resume
        preamble_idx = msg.index(COMPACT_CONTINUATION_PREAMBLE)
        summary_idx = msg.index("Summary:\nFull test")
        transcript_idx = msg.index("/conv/thread.md")
        recent_idx = msg.index(COMPACT_RECENT_MESSAGES_NOTE)
        resume_idx = msg.index(COMPACT_DIRECT_RESUME_INSTRUCTION)

        assert preamble_idx < summary_idx < transcript_idx < recent_idx < resume_idx


class TestPromptConstantsIntegrity:
    """Verify prompt constants haven't been accidentally truncated."""

    def test_summary_prompt_has_substantial_content(self):
        assert len(COMPACT_SUMMARY_PROMPT) > 1000

    def test_preamble_is_non_empty(self):
        assert len(COMPACT_CONTINUATION_PREAMBLE) > 50

    def test_resume_instruction_is_non_empty(self):
        assert len(COMPACT_DIRECT_RESUME_INSTRUCTION) > 50
