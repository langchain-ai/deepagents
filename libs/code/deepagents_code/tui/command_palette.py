"""Command palette providers for deepagents-code."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, AsyncIterator, Mapping, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider

from deepagents_code.model_config import (
    ModelProfileEntry,
    ModelSpec,
    get_available_models,
    get_model_profiles,
)
from deepagents_code.tui.widgets.model_selector import _RECOMMENDED_MODELS

if TYPE_CHECKING:
    from deepagents_code.app import DeepAgentsApp


def _get_display_name(model_spec: str, profiles: Mapping[str, ModelProfileEntry]) -> str:
    """Resolve the friendly display name for a model spec.

    Matches the logic in `ModelSelectorScreen._get_model_display_name`.
    """
    entry = profiles.get(model_spec)
    if entry:
        profile = entry.get("profile")
        if isinstance(profile, dict):
            name = profile.get("name")
            if isinstance(name, str) and name:
                return name

    recommended = _RECOMMENDED_MODELS.get(model_spec)
    if recommended:
        return recommended

    parsed = ModelSpec.try_parse(model_spec)
    if parsed and parsed.model:
        return parsed.model

    return model_spec


class ModelProvider(Provider):
    """A command palette provider for switching models."""

    async def search(self, query: str) -> Hits:
        """Search for models matching the query.

        Args:
            query: The search query.

        Yields:
            Fuzzy-matched model hits.
        """
        matcher = self.matcher(query)
        app = self.app

        # We need these methods to function. If they are missing (e.g. in some
        # test environments or if the provider is mounted on a different app),
        # we degrade gracefully by yielding nothing.
        switch_model = getattr(app, "_switch_model", None)
        effective_model_spec = getattr(app, "_effective_model_spec", None)

        if not (callable(switch_model) and callable(effective_model_spec)):
            return

        available = get_available_models()
        profiles = get_model_profiles()
        current_spec = effective_model_spec()

        for provider, models in available.items():
            for model in models:
                model_spec = f"{provider}:{model}"
                display_name = _get_display_name(model_spec, profiles)

                if model_spec == current_spec:
                    display_name = f"{display_name} (current)"

                score = matcher.match(display_name)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(display_name),
                        partial(switch_model, model_spec),
                        help=f"Switch to {display_name} ({provider})",
                    )

    async def discover(self) -> Hits:
        """Yield all models for discovery (empty query).

        Yields:
            All available model discovery hits.
        """
        app = self.app

        switch_model = getattr(app, "_switch_model", None)
        effective_model_spec = getattr(app, "_effective_model_spec", None)

        if not (callable(switch_model) and callable(effective_model_spec)):
            return

        available = get_available_models()
        profiles = get_model_profiles()
        current_spec = effective_model_spec()

        for provider, models in available.items():
            for model in models:
                model_spec = f"{provider}:{model}"
                display_name = _get_display_name(model_spec, profiles)

                if model_spec == current_spec:
                    display_name = f"{display_name} (current)"

                yield DiscoveryHit(
                    display_name,
                    partial(switch_model, model_spec),
                    help=f"Switch to {display_name} ({provider})",
                )
