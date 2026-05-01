"""Tests for the `_api/deprecation` adapter."""

import warnings
from collections.abc import Callable

import pytest

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

        def caller() -> None:
            deprecated_fn()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            caller()

        assert len(captured) == 1
        # `stacklevel=2` (default) attributes to the caller of `deprecated_fn`,
        # which is `caller` — defined in this test file.
        assert captured[0].filename == __file__

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

        def user_code() -> None:
            deprecated_init()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            user_code()

        assert len(captured) == 1
        assert captured[0].filename == __file__

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


def test_suppress_silences_warn_deprecated() -> None:
    """The re-exported context manager must suppress emissions from `warn_deprecated`."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with suppress_langchain_deprecation_warning():
            warn_deprecated(
                since="0.5.0",
                removal="1.0.0",
                message="suppressed",
                package="deepagents",
            )

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


# -- conftest drift detection -----------------------------------------------


def test_conftest_dedupe_targets_cover_known_deprecations() -> None:
    """Every `@deprecated`-decorated callable in deepagents must be in `_DEDUPED_TARGETS`.

    Catches drift: a future `@deprecated` addition without conftest update would
    re-introduce xdist reorder-flake before a single test exposed it.
    """
    target_names = {getattr(t.fget if isinstance(t, property) else t, "__qualname__", repr(t)) for t in _DEDUPED_TARGETS}

    expected_subset = {
        "BackendProtocol.ls_info",
        "BackendProtocol.als_info",
        "BackendProtocol.glob_info",
        "BackendProtocol.aglob_info",
        "BackendProtocol.grep_raw",
        "BackendProtocol.agrep_raw",
        "_NamespaceRuntimeCompat.runtime",
        "_NamespaceRuntimeCompat.state",
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
