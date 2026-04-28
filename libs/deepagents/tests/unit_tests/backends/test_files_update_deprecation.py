"""Tests for the deprecated ``files_update`` field on WriteResult and EditResult.

``files_update`` was removed in favour of internal backend state handling.
Passing it should emit a DeprecationWarning (scheduled for removal in 0.6.0)
but must not raise an error so existing callers aren't broken.
"""

import warnings

from deepagents.backends.protocol import EditResult, WriteResult

# -- WriteResult -----------------------------------------------------------


class TestWriteResultFilesUpdateDeprecation:
    def test_no_warning_without_files_update(self) -> None:
        """Normal construction without files_update should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = WriteResult(path="/f.txt")
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0
        assert result.path == "/f.txt"

    def test_warning_with_files_update_none(self) -> None:
        """Explicitly passing files_update=None should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = WriteResult(path="/f.txt", files_update=None)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "files_update" in str(deprecation_warnings[0].message)
        assert "0.6.0" in str(deprecation_warnings[0].message)
        assert result.files_update is None

    def test_warning_with_files_update_dict(self) -> None:
        """Explicitly passing files_update={...} should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WriteResult(path="/f.txt", files_update={"/f.txt": {"content": "x"}})
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "files_update" in str(deprecation_warnings[0].message)

    def test_error_field_still_works(self) -> None:
        """Error-only construction should still work fine."""
        result = WriteResult(error="File exists")
        assert result.error == "File exists"
        assert result.path is None


# -- EditResult ------------------------------------------------------------


class TestEditResultFilesUpdateDeprecation:
    def test_no_warning_without_files_update(self) -> None:
        """Normal construction without files_update should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EditResult(path="/f.txt", occurrences=1)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0
        assert result.path == "/f.txt"
        assert result.occurrences == 1

    def test_warning_with_files_update_none(self) -> None:
        """Explicitly passing files_update=None should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EditResult(path="/f.txt", files_update=None, occurrences=2)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "files_update" in str(deprecation_warnings[0].message)
        assert "0.6.0" in str(deprecation_warnings[0].message)
        assert result.files_update is None
        assert result.occurrences == 2

    def test_warning_with_files_update_dict(self) -> None:
        """Explicitly passing files_update={...} should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EditResult(path="/f.txt", files_update={"/f.txt": {"content": "x"}}, occurrences=1)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "files_update" in str(deprecation_warnings[0].message)
        assert result.occurrences == 1

    def test_error_field_still_works(self) -> None:
        """Error-only construction should still work fine."""
        result = EditResult(error="File not found")
        assert result.error == "File not found"
        assert result.path is None
        assert result.occurrences is None
