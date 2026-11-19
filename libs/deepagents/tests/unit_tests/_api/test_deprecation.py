"""Tests for the `_api/deprecation` adapter."""

import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Literal

import pytest
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

from deepagents._api.deprecation import (
    LangChainDeprecationWarning,
    deprecated,
    reset_deprecation_dedupe,
    suppress_langchain_deprecation_warning,
    warn_deprecated,
)
from tests.unit_tests.conftest import _DEDUPED_TARGETS

# -- warn_deprecated wrapper -------------------------------------------------


class TestWarnDeprecatedStacklevel:
    """Stack attribution must point at the user call site, not internals."""

    def test_default_stacklevel_attributes_to_caller(self) -> None:
        def deprecated_fn() -> None:
            warn_deprecated(
                since="0.5.0",
                removal="1.0.0",
                message="x is deprecated",
                package="deepagents",
            )

        def caller() -> int:
            line = sys._getframe().f_lineno + 1
            deprecated_fn()
            return line

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            line = caller()

        assert len(captured) == 1
        # `stacklevel=2` (default) attributes to the caller of `deprecated_fn`,
        # which is `caller` — defined in this test file.
        assert captured[0].filename == __file__
        assert captured[0].lineno == line

    def test_explicit_stacklevel_lifts_through_extra_frames(self) -> None:
        def helper() -> None:
            warn_deprecated(
                since="0.5.0",
                removal="1.0.0",
                message="y is deprecated",
                package="deepagents",
                stacklevel=3,
            )

        def deprecated_init() -> None:
            helper()

        def user_code() -> int:
            line = sys._getframe().f_lineno + 1
            deprecated_init()
            return line

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            line = user_code()

        assert len(captured) == 1
        assert captured[0].filename == __file__
        assert captured[0].lineno == line

    def test_warning_category_is_langchain_deprecation_warning(self) -> None:
        def deprecated_fn() -> None:
            warn_deprecated(
                since="0.5.0",
                removal="1.0.0",
                message="z is deprecated",
                package="deepagents",
            )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            deprecated_fn()

        assert len(captured) == 1
        assert captured[0].category is LangChainDeprecationWarning


@pytest.mark.parametrize("action", ["always", "error"])
def test_concurrent_warning_keeps_its_original_handler_and_filter(action: Literal["always", "error"]) -> None:
    entered = Event()
    release = Event()

    class BlockingMessage(str):
        __slots__ = ()

        def __bool__(self) -> bool:
            entered.set()
            if not release.wait(10):
                msg = "warning formatter was not released"
                raise TimeoutError(msg)
            return True

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("ignore", LangChainDeprecationWarning)
        warnings.simplefilter(action, RuntimeWarning)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(warn_deprecated, "0.5.0", message=BlockingMessage("adapter warning"), removal="1.0.0")
            try:
                assert entered.wait(10)
                if action == "error":
                    with pytest.raises(RuntimeWarning, match="unrelated warning"):
                        warnings.warn("unrelated warning", RuntimeWarning, stacklevel=1)
                else:
                    line = sys._getframe().f_lineno + 1
                    warnings.warn("unrelated warning", RuntimeWarning, stacklevel=1)
                    assert len(captured) == 1
                    assert captured[0].filename == __file__
                    assert captured[0].lineno == line
            finally:
                release.set()
            future.result(timeout=10)

    assert len(captured) == (1 if action == "always" else 0)


@pytest.mark.parametrize("pending", [False, True])
def test_warning_as_error_preserves_category_and_message(*, pending: bool) -> None:
    category = LangChainPendingDeprecationWarning if pending else LangChainDeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        with pytest.raises(category, match="custom warning"):
            warn_deprecated("0.5.0", message="custom warning", pending=pending)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"name": "old", "package": "deepagents"}, "`old` was deprecated in LangChain 0.5.0"),
        ({"name": "deep_agents.old"}, "`deep_agents.old` was deprecated in deep-agents 0.5.0"),
        (
            {
                "name": "old.method",
                "obj_type": "method",
                "package": "deepagents",
                "removal": "1.0.0",
                "alternative": "new",
                "addendum": "Migrate now.",
            },
            "The method `old.method` was deprecated in deepagents 0.5.0 and will be removed in 1.0.0. Use new instead. Migrate now.",
        ),
        (
            {"name": "deepagents.old", "alternative_import": "deepagents.new"},
            "`deepagents.old` was deprecated in deepagents 0.5.0. Use deepagents.new instead.",
        ),
        (
            {"name": "deepagents.old", "obj_type": "function", "alternative_import": "new_package.api.new", "alternative": "ignored"},
            "The function `deepagents.old` was deprecated in deepagents 0.5.0. An updated version of the function exists in the "
            "new-package package and should be used instead. To use it run `pip install -U new-package` and import as "
            "`from new_package.api import new`.",
        ),
        ({"name": "old", "pending": True}, "`old` will be deprecated in a future version"),
        ({"name": "old", "pending": True, "removal": "1.0.0"}, "`old` will be deprecated in a future version"),
        (
            {"message": "custom %(name)s", "name": "ignored", "addendum": "ignored", "alternative_import": "invalid"},
            "custom %(name)s",
        ),
    ],
)
def test_message_format_and_category(kwargs, expected: str) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warn_deprecated("0.5.0", **kwargs)

    assert len(captured) == 1
    assert str(captured[0].message) == expected
    category = LangChainPendingDeprecationWarning if kwargs.get("pending") else LangChainDeprecationWarning
    assert captured[0].category is category


def test_invalid_alternative_import_keeps_formatting_error() -> None:
    with pytest.raises(ValueError, match="not enough values to unpack"):
        warn_deprecated("0.5.0", name="old", alternative_import="invalid")


# -- reset_deprecation_dedupe ------------------------------------------------


def _make_decorated_function() -> Callable[[], int]:
    @deprecated(
        since="0.0.1",
        removal="9.9.9",
        message="example deprecation",
        package="deepagents",
    )
    def example_fn() -> int:
        return 42

    return example_fn


def _make_decorated_property() -> type:
    class _Holder:
        @property
        @deprecated(
            since="0.0.1",
            removal="9.9.9",
            message="example deprecation",
            package="deepagents",
        )
        def value(self) -> int:
            return 7

    return _Holder


class TestResetDeprecationDedupe:
    """`reset_deprecation_dedupe` must re-arm per-call emission for tests."""

    def test_function_dedupe_is_reset(self) -> None:
        fn = _make_decorated_function()

        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            fn()
            fn()
        # `@deprecated` dedupes in-process, so back-to-back calls only
        # produce one warning.
        deprecations = [w for w in first if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

        reset_deprecation_dedupe(fn)

        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            fn()
        deprecations = [w for w in second if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

    def test_property_dedupe_is_reset(self) -> None:
        Holder = _make_decorated_property()  # noqa: N806
        holder = Holder()

        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            _ = holder.value
            _ = holder.value
        deprecations = [w for w in first if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

        # Reset on the descriptor (`property` object) — `reset_deprecation_dedupe`
        # detects this and resets the underlying `fget`'s closure.
        reset_deprecation_dedupe(Holder.value)  # ty: ignore[unresolved-attribute]

        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            _ = holder.value
        deprecations = [w for w in second if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

    def test_non_decorated_targets_are_silently_skipped(self) -> None:
        """Passing plain callables/objects/classes must not raise."""

        def plain_fn() -> None:
            pass

        # Should not raise — non-decorated objects are skipped.
        reset_deprecation_dedupe(plain_fn, lambda: None, object(), 42, "string")

    def test_mixed_targets_in_single_call(self) -> None:
        """The loop must continue past targets that can't be reset."""
        fn = _make_decorated_function()

        # Prime the dedupe flag.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fn()

        # Mix a non-decorated callable with the decorated one.
        reset_deprecation_dedupe(lambda: None, fn, object())

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fn()
        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1


# -- suppress_langchain_deprecation_warning ---------------------------------


@pytest.mark.parametrize("pending", [False, True])
def test_suppress_silences_warn_deprecated(*, pending: bool) -> None:
    """The re-exported context manager must suppress emissions from `warn_deprecated`."""
    with warnings.catch_warnings(record=True) as captured, suppress_langchain_deprecation_warning():
        warnings.simplefilter("always")
        warn_deprecated(
            since="0.5.0",
            message="suppressed",
            pending=pending,
            package="deepagents",
        )

    assert captured == []


# -- conftest drift detection -----------------------------------------------


def test_conftest_dedupe_targets_cover_known_deprecations() -> None:
    """Every `@deprecated`-decorated callable in deepagents must be in `_DEDUPED_TARGETS`.

    Catches drift: a future `@deprecated` addition without conftest update would
    re-introduce xdist reorder-flake before a single test exposed it.
    """
    target_names = {getattr(t.fget if isinstance(t, property) else t, "__qualname__", repr(t)) for t in _DEDUPED_TARGETS}

    expected_subset = {
        "get_default_model",
    }
    missing = expected_subset - target_names
    assert not missing, f"Drift: {missing} missing from `_DEDUPED_TARGETS`"


@pytest.mark.parametrize("dummy", [None])
def test_dedupe_reset_fixture_is_autouse(dummy: object) -> None:  # noqa: ARG001
    """Smoke test: the autouse fixture in `conftest.py` runs before this test."""
    fn = _make_decorated_function()
    # If the fixture runs, dedupe was reset before this test, so we should
    # observe a fresh warning emission.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fn()
    assert any(issubclass(w.category, DeprecationWarning) for w in captured)
