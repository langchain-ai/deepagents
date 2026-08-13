"""Tests for Inworld LLM router catalog discovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import patch

import pytest

from deepagents_code import inworld_catalog

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> Iterator[None]:
    """Clear the module-level catalog cache before and after each test."""
    inworld_catalog.clear_cache()
    yield
    inworld_catalog.clear_cache()


def _entry(
    provider: str = "openai",
    model: str = "gpt-5.2",
    **spec: Any,
) -> dict[str, Any]:
    """Build a catalog entry with an overridable `spec`."""
    base: dict[str, Any] = {
        "contextLength": 272000,
        "maxCompletionTokens": 128000,
        "inputModalities": ["text", "image"],
        "outputModalities": ["text"],
        "capabilities": {"functionCalling": True, "reasoning": True},
    }
    base.update(spec)
    return {
        "model": model,
        "provider": provider,
        "isSupported": True,
        "spec": base,
    }


class _FakeResponse:
    """Minimal `urlopen` context manager returning a fixed body."""

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        """Return the encoded response body."""
        return self._body


class TestCatalogUrl:
    """Tests for deriving the catalog endpoint from a completions base URL."""

    def test_defaults_to_public_router(self) -> None:
        """`None` resolves against the default base URL."""
        assert (
            inworld_catalog.catalog_url(None)
            == "https://api.inworld.ai/llm/v1alpha/models"
        )

    def test_replaces_versioned_path(self) -> None:
        """The catalog path replaces the `/v1` path segment."""
        assert (
            inworld_catalog.catalog_url("https://api.inworld.ai/v1")
            == "https://api.inworld.ai/llm/v1alpha/models"
        )

    def test_follows_custom_host(self) -> None:
        """A custom base URL yields a catalog URL on the same host."""
        assert (
            inworld_catalog.catalog_url("https://proxy.internal/v1")
            == "https://proxy.internal/llm/v1alpha/models"
        )


class TestParseCatalog:
    """Tests for turning a catalog payload into model profiles."""

    def test_maps_spec_fields_to_profile_keys(self) -> None:
        """Spec fields map to model-profile keys."""
        catalog = inworld_catalog.parse_catalog({"models": [_entry()]})

        assert catalog == {
            "openai/gpt-5.2": {
                "max_input_tokens": 272000,
                "max_output_tokens": 128000,
                "text_inputs": True,
                "image_inputs": True,
                "audio_inputs": False,
                "text_outputs": True,
                "tool_calling": True,
                "reasoning_output": True,
            }
        }

    def test_unmapped_modalities_stay_unstated(self) -> None:
        """Unmapped modalities are omitted from the profile."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(inputModalities=["text", "video"])]}
        )

        profile = catalog["openai/gpt-5.2"]
        assert "pdf_inputs" not in profile
        assert "video_inputs" not in profile

    def test_model_id_joins_provider_and_model(self) -> None:
        """Model ids join provider and model with a slash."""
        catalog = inworld_catalog.parse_catalog(
            {
                "models": [
                    _entry(provider="deepinfra", model="deepseek-ai/DeepSeek-V3.2"),
                    _entry(provider="inworld", model="models/gemma-4-31b-it"),
                ]
            }
        )

        assert set(catalog) == {
            "deepinfra/deepseek-ai/DeepSeek-V3.2",
            "inworld/models/gemma-4-31b-it",
        }

    def test_first_party_models_get_an_unprefixed_display_name(self) -> None:
        """First-party models get a `name` without the `models/` prefix."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(provider="inworld", model="models/GLM-5.2")]}
        )

        assert catalog["inworld/models/GLM-5.2"]["name"] == "GLM-5.2"

    def test_upstream_models_keep_their_qualified_id_as_label(self) -> None:
        """Upstream models get no `name` override."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(provider="deepinfra", model="zai-org/GLM-5.2")]}
        )

        assert "name" not in catalog["deepinfra/zai-org/GLM-5.2"]

    def test_reasoning_levels_become_effort_labels(self) -> None:
        """`EFFORT_*` levels become lowercase labels in catalog order."""
        catalog = inworld_catalog.parse_catalog(
            {
                "models": [
                    _entry(
                        capabilities={
                            "reasoning": True,
                            "reasoningCapability": {
                                "supported": True,
                                "supportedLevels": [
                                    "EFFORT_NONE",
                                    "EFFORT_LOW",
                                    "EFFORT_XHIGH",
                                ],
                            },
                        }
                    )
                ]
            }
        )

        assert catalog["openai/gpt-5.2"]["reasoning_effort_levels"] == [
            "none",
            "low",
            "xhigh",
        ]

    def test_unfamiliar_levels_are_passed_through(self) -> None:
        """Unrecognized effort levels are passed through."""
        catalog = inworld_catalog.parse_catalog(
            {
                "models": [
                    _entry(
                        capabilities={
                            "reasoningCapability": {
                                "supportedLevels": ["EFFORT_TURBO_V2"]
                            }
                        }
                    )
                ]
            }
        )

        assert catalog["openai/gpt-5.2"]["reasoning_effort_levels"] == ["turbo_v2"]

    def test_unsupported_reasoning_yields_no_levels(self) -> None:
        """`supported: false` yields no effort levels."""
        catalog = inworld_catalog.parse_catalog(
            {
                "models": [
                    _entry(
                        capabilities={
                            "reasoningCapability": {
                                "supported": False,
                                "supportedLevels": ["EFFORT_LOW"],
                            }
                        }
                    )
                ]
            }
        )

        assert "reasoning_effort_levels" not in catalog["openai/gpt-5.2"]

    @pytest.mark.parametrize(
        "capability",
        [
            {},
            {"reasoningCapability": None},
            {"reasoningCapability": {"supportedLevels": "EFFORT_LOW"}},
            {"reasoningCapability": {"supportedLevels": []}},
            {"reasoningCapability": {"supportedLevels": [7, ""]}},
        ],
    )
    def test_malformed_reasoning_capability_yields_no_levels(
        self, capability: dict[str, Any]
    ) -> None:
        """A malformed reasoning capability yields no effort levels."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(capabilities=capability)]}
        )

        assert "reasoning_effort_levels" not in catalog["openai/gpt-5.2"]

    def test_omits_capabilities_the_catalog_does_not_state(self) -> None:
        """Capabilities the entry omits are absent from the profile."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(capabilities={"vision": True})]}
        )

        profile = catalog["openai/gpt-5.2"]
        assert "tool_calling" not in profile
        assert "reasoning_output" not in profile

    def test_reports_tool_calling_false_when_stated(self) -> None:
        """`functionCalling: false` maps to `tool_calling: False`."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(capabilities={"functionCalling": False})]}
        )

        assert catalog["openai/gpt-5.2"]["tool_calling"] is False

    def test_skips_unsupported_and_malformed_entries(self) -> None:
        """Unsupported, non-dict, and id-less entries are dropped."""
        unsupported = _entry(model="retired")
        unsupported["isSupported"] = False
        missing_provider = _entry(model="orphan")
        del missing_provider["provider"]

        catalog = inworld_catalog.parse_catalog(
            {"models": [unsupported, missing_provider, "junk", _entry()]}
        )

        assert set(catalog) == {"openai/gpt-5.2"}

    def test_rejects_non_positive_token_counts(self) -> None:
        """Non-positive token counts are omitted."""
        catalog = inworld_catalog.parse_catalog(
            {"models": [_entry(contextLength=0, maxCompletionTokens=-1)]}
        )

        profile = catalog["openai/gpt-5.2"]
        assert "max_input_tokens" not in profile
        assert "max_output_tokens" not in profile

    @pytest.mark.parametrize("payload", [None, [], {}, {"models": "nope"}])
    def test_tolerates_unexpected_payload_shapes(self, payload: object) -> None:
        """Unexpected payload shapes yield an empty mapping."""
        assert inworld_catalog.parse_catalog(payload) == {}


class TestModelProfiles:
    """Tests for the cached, best-effort discovery entry point."""

    def test_no_credential_skips_the_probe(self) -> None:
        """No credential skips the HTTP request."""
        with patch.object(inworld_catalog, "urlopen") as mock_urlopen:
            catalog = inworld_catalog.model_profiles(None, None)

        mock_urlopen.assert_not_called()
        assert catalog == {inworld_catalog.ROUTER_MODEL: {"tool_calling": True}}

    def test_transport_failure_falls_back_to_router_model(self) -> None:
        """A transport failure returns the router-only catalog."""
        with patch.object(inworld_catalog, "urlopen", side_effect=OSError("boom")):
            catalog = inworld_catalog.model_profiles(None, "key")

        assert catalog == {inworld_catalog.ROUTER_MODEL: {"tool_calling": True}}

    def test_discovered_models_include_the_router_entry(self) -> None:
        """Discovered models are returned alongside the router entry."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ):
            catalog = inworld_catalog.model_profiles(None, "key")

        assert list(catalog) == [inworld_catalog.ROUTER_MODEL, "openai/gpt-5.2"]
        assert catalog["openai/gpt-5.2"]["max_input_tokens"] == 272000

    def test_sends_documented_basic_scheme(self) -> None:
        """The credential is sent with the `Basic` scheme."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": []}),
        ) as mock_urlopen:
            inworld_catalog.model_profiles(None, "secret")

        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") == "Basic secret"

    def test_successful_catalog_is_cached(self) -> None:
        """A successful catalog is fetched once and reused."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ) as mock_urlopen:
            first = inworld_catalog.model_profiles(None, "key")
            second = inworld_catalog.model_profiles(None, "key")

        assert mock_urlopen.call_count == 1
        assert first == second

    def test_failed_catalog_is_not_cached(self) -> None:
        """A failed fetch is not cached."""
        with patch.object(inworld_catalog, "urlopen", side_effect=OSError("boom")):
            inworld_catalog.model_profiles(None, "key")

        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ):
            catalog = inworld_catalog.model_profiles(None, "key")

        assert "openai/gpt-5.2" in catalog

    def test_clear_cache_forces_a_refetch(self) -> None:
        """`clear_cache` forces the next lookup to re-fetch."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ) as mock_urlopen:
            inworld_catalog.model_profiles(None, "key")
            inworld_catalog.clear_cache()
            inworld_catalog.model_profiles(None, "key")

        assert mock_urlopen.call_count == 2

    def test_cache_is_keyed_by_endpoint(self) -> None:
        """The cache is keyed by catalog URL."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ) as mock_urlopen:
            inworld_catalog.model_profiles("https://api.inworld.ai/v1", "key")
            inworld_catalog.model_profiles("https://proxy.internal/v1", "key")

        assert mock_urlopen.call_count == 2

    def test_callers_cannot_mutate_the_cache(self) -> None:
        """Returned profiles are copies of the cached entries."""
        with patch.object(
            inworld_catalog,
            "urlopen",
            return_value=_FakeResponse({"models": [_entry()]}),
        ):
            first = inworld_catalog.model_profiles(None, "key")
            first["openai/gpt-5.2"]["max_input_tokens"] = 1
            second = inworld_catalog.model_profiles(None, "key")

        assert second["openai/gpt-5.2"]["max_input_tokens"] == 272000

    def test_non_http_endpoint_is_refused(self) -> None:
        """A schemeless endpoint skips the request."""
        with patch.object(inworld_catalog, "urlopen") as mock_urlopen:
            catalog = inworld_catalog.model_profiles("api.inworld.ai/v1", "key")

        mock_urlopen.assert_not_called()
        assert catalog == {inworld_catalog.ROUTER_MODEL: {"tool_calling": True}}
