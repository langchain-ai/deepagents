"""Live model-catalog discovery for the Inworld LLM router.

Inworld serves its own models alongside many upstream providers (`inworld/`,
`openai/`, `anthropic/`, `deepinfra/`, ...) behind one OpenAI-compatible
endpoint, and ships no LangChain package to carry their profiles.

Discovery is best-effort: any failure degrades to `ROUTER_MODEL` alone, which
routes server-side and so works without a catalog.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, cast
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PROVIDER = "inworld"
"""Provider name for the Inworld LLM router."""

DEFAULT_BASE_URL = "https://api.inworld.ai/v1"
"""Default OpenAI-compatible completions endpoint for the router."""

CATALOG_PATH = "/llm/v1alpha/models"
"""Host-absolute path of the router's model catalog, resolved against `base_url`."""

FIRST_PARTY_MODEL_PREFIX = "models/"
"""Prefix the catalog gives Inworld's own models (e.g. `models/GLM-5.2`).

Stripped from the display name only, never from the id: the short form the API
echoes back (`inworld/X`) is rejected as a request id, and the bare name is
ambiguous across upstreams.
"""

ROUTER_MODEL = "auto"
"""Model id that delegates model selection to the router itself.

Claims tool calling so the selector's capability filter keeps it.
"""

DISCOVERY_TIMEOUT_SECONDS = 5.0
"""Socket timeout for catalog requests."""

_EFFORT_LEVEL_PREFIX = "EFFORT_"
"""Prefix on the catalog's reasoning levels (`EFFORT_LOW`, `EFFORT_XHIGH`, ...).

Stripped and lowercased to reach the app's effort vocabulary, which treats
labels as opaque -- so a level Inworld adds later needs no code change.
"""

_MODALITY_PROFILE_KEYS: dict[str, str] = {
    "text": "text_inputs",
    "image": "image_inputs",
    "audio": "audio_inputs",
}
"""Catalog `inputModalities` entries mapped to model-profile keys.

`inputModalities` is exhaustive per model, so one it omits is marked
unsupported. Modalities left out of this map stay unstated rather than
asserted `False`.
"""

_catalog_cache: dict[str, dict[str, dict[str, Any]]] = {}
_catalog_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop cached catalogs so the next lookup re-fetches."""
    with _catalog_cache_lock:
        _catalog_cache.clear()


def catalog_url(base_url: str | None) -> str:
    """Return the catalog endpoint that pairs with a completions `base_url`.

    The catalog sits outside the `/v1` prefix, so its path replaces rather than
    extends `base_url`'s -- while still following a custom host to its own
    catalog.

    Args:
        base_url: Completions endpoint, or `None` for `DEFAULT_BASE_URL`.

    Returns:
        Absolute URL of the catalog endpoint.
    """
    return urljoin(base_url or DEFAULT_BASE_URL, CATALOG_PATH)


def _coerce_positive_int(value: object) -> int | None:
    """Return `value` as a positive int, or `None` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _profile_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract model-profile fields from one catalog entry.

    Args:
        entry: A single object from the catalog's `models` array.

    Returns:
        Profile fields understood by the model selector. A capability the entry
            does not state is omitted rather than guessed, so it stays unknown
            instead of being asserted as `False`.
    """
    spec = entry.get("spec")
    if not isinstance(spec, dict):
        return {}

    profile: dict[str, Any] = {}

    context_length = _coerce_positive_int(spec.get("contextLength"))
    if context_length is not None:
        profile["max_input_tokens"] = context_length

    max_completion = _coerce_positive_int(spec.get("maxCompletionTokens"))
    if max_completion is not None:
        profile["max_output_tokens"] = max_completion

    input_modalities = spec.get("inputModalities")
    if isinstance(input_modalities, list):
        for modality, profile_key in _MODALITY_PROFILE_KEYS.items():
            profile[profile_key] = modality in input_modalities

    output_modalities = spec.get("outputModalities")
    if isinstance(output_modalities, list):
        profile["text_outputs"] = "text" in output_modalities

    capabilities = spec.get("capabilities")
    if isinstance(capabilities, dict):
        function_calling = capabilities.get("functionCalling")
        if isinstance(function_calling, bool):
            profile["tool_calling"] = function_calling
        reasoning = capabilities.get("reasoning")
        if isinstance(reasoning, bool):
            profile["reasoning_output"] = reasoning
        levels = _effort_levels(capabilities.get("reasoningCapability"))
        if levels:
            profile["reasoning_effort_levels"] = levels

    return profile


def _effort_levels(reasoning_capability: object) -> list[str]:
    """Extract selectable reasoning effort levels from a catalog capability.

    Without these the effort picker has nothing to offer and the model runs at
    whatever the router defaults to.

    Args:
        reasoning_capability: The entry's `capabilities.reasoningCapability`.

    Returns:
        Effort labels in catalog order, or an empty list when the model states
            no levels or reports reasoning as unsupported.
    """
    if not isinstance(reasoning_capability, dict):
        return []
    if reasoning_capability.get("supported") is False:
        return []
    levels = reasoning_capability.get("supportedLevels")
    if not isinstance(levels, list):
        return []
    return [
        level.removeprefix(_EFFORT_LEVEL_PREFIX).lower()
        for level in levels
        if isinstance(level, str) and level
    ]


def parse_catalog(payload: object) -> dict[str, dict[str, Any]]:
    """Build a `{model_id: profile}` mapping from a catalog payload.

    Ids are `<provider>/<model>`, the exact string the completions endpoint
    expects. First-party models also get a short `name`; upstream models keep
    their qualified id, which is what distinguishes the same model served by
    several upstreams.

    Args:
        payload: Decoded JSON body of a catalog response.

    Returns:
        Mapping of model id to profile fields. Entries the router reports as
            unsupported are skipped, as are malformed ones.
    """
    if not isinstance(payload, dict):
        return {}
    models = payload.get("models")
    if not isinstance(models, list):
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("isSupported") is False:
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        if not isinstance(provider, str) or not provider:
            continue
        if not isinstance(model, str) or not model:
            continue
        profile = _profile_from_entry(cast("dict[str, Any]", entry))
        if provider == PROVIDER:
            profile["name"] = model.removeprefix(FIRST_PARTY_MODEL_PREFIX)
        catalog[f"{provider}/{model}"] = profile
    return catalog


def _fetch_catalog(
    url: str,
    api_key: str,
    *,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Fetch and parse the router catalog.

    Sent with the `Basic` scheme Inworld documents. The completions path uses
    `Bearer` because the OpenAI-compatible client builds that header itself;
    Inworld accepts both.

    Args:
        url: Catalog endpoint.
        api_key: Credential for the `Authorization` header.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed catalog, or an empty mapping on any failure.
    """
    if not url.startswith(("http://", "https://")):
        logger.warning(
            "Skipping Inworld catalog discovery: %r has no http:// or https:// "
            "scheme. Set base_url or INWORLD_BASE_URL to e.g. %s.",
            url,
            DEFAULT_BASE_URL,
        )
        return {}

    request = Request(  # noqa: S310  # scheme guarded above
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Basic {api_key}",
        },
    )
    # Broad because `pytest-socket`'s `SocketBlockedError` is not an `OSError`;
    # the split only sets the log level.
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310  # scheme guarded above
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        logger.debug("Inworld catalog discovery failed for %s: %s", url, exc)
        return {}
    except Exception as exc:  # noqa: BLE001  # see comment above
        logger.warning(
            "Inworld catalog discovery raised unexpected %s for %s: %s",
            type(exc).__name__,
            url,
            exc,
        )
        return {}

    catalog = parse_catalog(payload)
    if not catalog:
        logger.debug(
            "Inworld catalog at %s returned no usable models; payload shape was %s",
            url,
            type(payload).__name__,
        )
    return catalog


def model_profiles(
    base_url: str | None,
    api_key: str | None,
    *,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Return discovered Inworld models and their profiles.

    Cached for the process lifetime, keyed by catalog URL; `/reload` refreshes
    it via `model_config.clear_caches`. `ROUTER_MODEL` is always present, so an
    unreachable catalog still leaves the provider usable.

    Args:
        base_url: Completions endpoint, or `None` for `DEFAULT_BASE_URL`.
        api_key: Credential for the catalog request. Discovery is skipped when
            absent -- the endpoint requires auth and would only 401.
        timeout: Socket timeout in seconds.

    Returns:
        Mapping of model id to profile fields, ordered with `ROUTER_MODEL`
            first.
    """
    url = catalog_url(base_url)
    with _catalog_cache_lock:
        cached = _catalog_cache.get(url)
        if cached is not None:
            return {model: dict(profile) for model, profile in cached.items()}

    if api_key:
        discovered = _fetch_catalog(url, api_key, timeout=timeout)
    else:
        logger.debug(
            "Skipping Inworld catalog discovery for %s: no credential available",
            url,
        )
        discovered = {}

    catalog: dict[str, dict[str, Any]] = {ROUTER_MODEL: {"tool_calling": True}}
    catalog.update(discovered)

    if discovered:
        with _catalog_cache_lock:
            _catalog_cache[url] = {
                model: dict(profile) for model, profile in catalog.items()
            }
    return catalog
