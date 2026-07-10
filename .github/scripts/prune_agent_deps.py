"""Prune the Harbor LangGraph agent's provider deps to the active model.

The agent (`deepagents_harbor/langgraph_project/langgraph_agent.py`) picks its
chat model at runtime through `langchain.chat_models.init_chat_model`, which
lazily imports only the provider package matching the model spec's `provider:`
prefix. The committed `langgraph.json` therefore lists *every* provider so a
local `langgraph dev` can run any model — but a single CI job only ever runs one
model, and installing the other providers just slows the sandbox env build.

This script rewrites a `langgraph.json` in place, dropping every provider
package except the one matching `HARBOR_MODEL`'s provider. Non-provider
dependencies (langchain core, the staged local packages, MCP adapters, etc.)
are always kept. It runs in CI against the ephemeral checkout
(`.github/workflows/harbor.yml`); the committed file is left untouched.

Usage:
    HARBOR_MODEL=fireworks:accounts/... \\
        python prune_agent_deps.py path/to/langgraph.json
"""

from __future__ import annotations

import json
import os
import sys

PROVIDER_TO_PACKAGE: dict[str, str] = {
    "anthropic": "langchain-anthropic",
    "baseten": "langchain-baseten",
    "fireworks": "langchain-fireworks",
    "google_genai": "langchain-google-genai",
    "groq": "langchain-groq",
    "nvidia": "langchain-nvidia-ai-endpoints",
    "ollama": "langchain-ollama",
    "openai": "langchain-openai",
    "openrouter": "langchain-openrouter",
    "xai": "langchain-xai",
}
"""Model-spec provider prefix -> pip package that supplies it.

Must stay in sync with the provider packages listed in `langgraph.json`. Keep
this hardcoded (not derived): it is the single source of truth for which
packages are prunable and what each provider maps to. The `prune_dependencies`
drift guard fails the run if the matched package is missing from the file, so a
stale entry here surfaces loudly rather than silently shipping an agent env with
no provider.
"""

PRUNABLE_PACKAGES: frozenset[str] = frozenset(PROVIDER_TO_PACKAGE.values())
"""Provider packages subject to pruning.

A dependency is removed only when its package name is in this set *and* does not
match the active provider; every other dependency (langchain core, local path
deps, MCP adapters, aiohttp, toml, ...) is kept verbatim.
"""

_NAME_DELIMITERS = ("<", ">", "=", "!", "~", " ", "[", ";", ",")
"""Characters that terminate the package-name head of a PEP 508 requirement.

Splitting on these (no regex) recovers the bare package name from a spec like
`langchain-openai>=1.3.0,<1.4.0`, `pkg[extra]`, or `pkg; python_version<'3.13'`.
"""


def dependency_package(dep: str) -> str:
    """Return the bare package name at the head of a dependency string.

    Truncates at the first version specifier, extra, or marker. Local path
    deps (e.g. `./.local_deps/deepagents`) contain no delimiter and are
    returned unchanged; they are never in `PRUNABLE_PACKAGES`, so they are
    always kept.
    """
    head = dep.strip()
    for sep in _NAME_DELIMITERS:
        head = head.split(sep, 1)[0]
    return head.strip()


def prune_dependencies(deps: list[str], provider: str) -> list[str]:
    """Return `deps` with every provider package removed except `provider`'s.

    Args:
        deps: The `dependencies` array from a `langgraph.json`.
        provider: Model-spec provider prefix (e.g. `fireworks`). Must be a key
            of `PROVIDER_TO_PACKAGE`.

    Returns:
        The filtered dependency list, in the original order.

    Raises:
        KeyError: If `provider` is not a known provider.
        ValueError: If the provider's package is absent from `deps` — a drift
            between `PROVIDER_TO_PACKAGE` and `langgraph.json` that would
            otherwise ship an agent env with no usable provider.
    """
    keep = PROVIDER_TO_PACKAGE[provider]
    kept = [
        dep
        for dep in deps
        if dependency_package(dep) not in PRUNABLE_PACKAGES
        or dependency_package(dep) == keep
    ]
    if not any(dependency_package(dep) == keep for dep in kept):
        msg = (
            f"Expected provider package {keep!r} for provider {provider!r} not "
            "found in langgraph.json dependencies; PROVIDER_TO_PACKAGE is out of "
            "sync with the agent config."
        )
        raise ValueError(msg)
    return kept


def main() -> None:
    """Rewrite the langgraph.json at argv[1], pruning to HARBOR_MODEL's provider."""
    if len(sys.argv) != 2:  # noqa: PLR2004
        raise SystemExit(f"Usage: {sys.argv[0]} <path-to-langgraph.json>")

    path = sys.argv[1]
    model = os.environ.get("HARBOR_MODEL", "").strip()
    if not model or ":" not in model:
        raise SystemExit(
            f"::error::HARBOR_MODEL must be a 'provider:model' spec (got {model!r})"
        )
    provider = model.split(":", 1)[0]

    with open(path, encoding="utf-8") as f:  # noqa: PTH123
        config = json.load(f)
    deps = config.get("dependencies", [])

    if provider not in PROVIDER_TO_PACKAGE:
        # An unmapped provider is non-functional in this workflow anyway:
        # harbor.yml only wires credentials and agent env for the providers in
        # PROVIDER_TO_PACKAGE, and langgraph.json ships only their packages.
        # Leaving deps in place would keep langchain-fireworks (a prerelease
        # dependency) and fail the agent-env install with a cryptic resolver
        # error; fail fast here with an actionable one instead.
        supported = ", ".join(sorted(PROVIDER_TO_PACKAGE))
        raise SystemExit(
            f"::error::Unsupported model provider {provider!r} (from {model!r}). "
            f"Supported providers: {supported}. To add one, extend "
            "PROVIDER_TO_PACKAGE and wire its package into langgraph.json plus "
            "the credential and agent-env steps in harbor.yml."
        )

    pruned = prune_dependencies(deps, provider)
    config["dependencies"] = pruned

    with open(path, "w", encoding="utf-8") as f:  # noqa: PTH123
        json.dump(config, f, indent=2)
        f.write("\n")

    removed = [dependency_package(d) for d in deps if d not in pruned]
    print(  # noqa: T201
        f"Pruned agent provider deps for {provider!r}: kept "
        f"{PROVIDER_TO_PACKAGE[provider]}"
        + (f", removed {len(removed)}: {', '.join(removed)}" if removed else "")
    )


if __name__ == "__main__":
    main()
