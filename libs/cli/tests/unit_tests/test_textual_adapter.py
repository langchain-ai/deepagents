"""Unit tests for textual_adapter functions."""

from deepagents_cli.textual_adapter import _is_summarization_chunk


class TestIsSummarizationChunk:
    """Tests for `_is_summarization_chunk` detection."""

    def test_returns_true_for_summarization_source(self) -> None:
        """Should return `True` when `lc_source` is `'summarization'`."""
        metadata = {"lc_source": "summarization"}
        assert _is_summarization_chunk(metadata) is True

    def test_returns_false_for_none_metadata(self) -> None:
        """Should return `False` when `metadata` is `None`."""
        assert _is_summarization_chunk(None) is False
        assert _is_summarization_chunk({}) is False

    def test_returns_false_for_none_lc_source(self) -> None:
        """Should return `False` when `lc_source` is not `'summarization'`."""
        metadata_none = {"lc_source": None}
        assert _is_summarization_chunk(metadata_none) is False

        metadata_other = {"lc_source": "other"}
        assert _is_summarization_chunk(metadata_other) is False

        metadata_missing = {"other_key": "value"}
        assert _is_summarization_chunk(metadata_missing) is False
