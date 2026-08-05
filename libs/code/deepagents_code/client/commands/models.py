"""List the model catalog that dcode can use."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def build_model_catalog() -> dict:
    """Build a credential-safe, machine-readable model catalog.

    Returns:
        Providers, model identifiers, defaults, and credential metadata.
    """
    from deepagents_code.model_config import (
        ModelConfig,
        get_available_models,
        get_provider_auth_status,
    )

    config = ModelConfig.load()
    providers = []
    for provider, models in get_available_models().items():
        status = get_provider_auth_status(provider)
        providers.append(
            {
                "id": provider,
                "models": list(models),
                "auth": {
                    "state": status.state.value,
                    "source": (
                        status.source.value if status.source is not None else None
                    ),
                    "env_var": status.env_var,
                    "detail": status.detail,
                },
            }
        )

    return {
        "default_model": config.default_model,
        "recent_model": config.recent_model,
        "providers": providers,
    }


def run_models_command(args: argparse.Namespace) -> int:
    """Print the available model catalog and return an exit code.

    Returns:
        Zero after the catalog is written.
    """
    catalog = build_model_catalog()
    if getattr(args, "output_format", "text") == "json":
        from deepagents_code.output import write_json

        write_json("models", catalog)
        return 0

    from deepagents_code.config import console

    default_model = catalog["default_model"] or catalog["recent_model"]
    for provider in catalog["providers"]:
        console.print(f"[bold]{provider['id']}[/bold]")
        for model in provider["models"]:
            marker = " *" if f"{provider['id']}:{model}" == default_model else ""
            console.print(f"  {model}{marker}")
    return 0
