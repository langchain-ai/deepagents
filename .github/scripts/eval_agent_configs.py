"""Shared agent-configuration registry for Unified Harbor evaluations."""

from __future__ import annotations

import argparse
import re
from typing import TypedDict


class RuntimeConfig(TypedDict):
    """Concrete Harbor graph and product packages for one runtime harness."""

    graph: str
    label: str
    packages: tuple[str, ...]


class CodeConfig(TypedDict):
    """Selectable code harness plus its optional conversation override."""

    runtime: str
    conversation_runtime: str | None


DEFAULT_CODE_CONFIG = "bare"
DEFAULT_CONVERSATION_RUNTIME = "tau3"

# Concrete implementations understood by the fixed evaluation controller. A future
# config-specific tau3 adapter is registered here and referenced through
# `conversation_runtime`; ordinary code harnesses intentionally share `tau3`.
RUNTIME_CONFIGS: dict[str, RuntimeConfig] = {
    "bare": {
        "graph": "bare_deepagent",
        "label": "bare create_deep_agent",
        "packages": ("deepagents",),
    },
    "dcode": {
        "graph": "deepagent",
        "label": "dcode harness",
        "packages": ("deepagents", "deepagents-code"),
    },
    "tau3": {
        "graph": "tau3_deepagent",
        "label": "tau3 conversational deepagent",
        "packages": ("deepagents",),
    },
}

CODE_CONFIGS: dict[str, CodeConfig] = {
    "bare": {"runtime": "bare", "conversation_runtime": None},
    "dcode": {"runtime": "dcode", "conversation_runtime": None},
}


def parse_code_configs(raw: str) -> list[str]:
    """Parse an order-preserving comma-separated selectable config list.

    Args:
        raw: Comma-separated config identifiers. Blank selects the default.

    Returns:
        Validated, de-duplicated config identifiers.

    Raises:
        ValueError: If any identifier is not registered as selectable.
    """
    configs = list(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    ) or [DEFAULT_CODE_CONFIG]
    unknown = [config for config in configs if config not in CODE_CONFIGS]
    if unknown:
        msg = f"agent configs must be in {sorted(CODE_CONFIGS)}, got unknown {unknown}"
        raise ValueError(msg)
    return configs


def runtime_for_code_config(config: str) -> str:
    """Return the autonomous/context runtime for a selectable config."""
    try:
        return CODE_CONFIGS[config]["runtime"]
    except KeyError as exc:
        msg = f"unknown code config: {config!r}"
        raise ValueError(msg) from exc


def conversation_runtime_for(config: str) -> str:
    """Return a config's conversation override or the shared tau3 runtime."""
    try:
        override = CODE_CONFIGS[config]["conversation_runtime"]
    except KeyError as exc:
        msg = f"unknown code config: {config!r}"
        raise ValueError(msg) from exc
    return override or DEFAULT_CONVERSATION_RUNTIME


def runtime_config(runtime: str) -> RuntimeConfig:
    """Return metadata for one concrete Harbor runtime implementation."""
    try:
        return RUNTIME_CONFIGS[runtime]
    except KeyError as exc:
        msg = f"unknown runtime config: {runtime!r}"
        raise ValueError(msg) from exc


def required_packages(runtimes: list[str]) -> list[str]:
    """Return the ordered union of product distributions used by runtimes."""
    packages: list[str] = []
    for runtime in runtimes:
        for package in runtime_config(runtime)["packages"]:
            if package not in packages:
                packages.append(package)
    return packages


def validate_registry() -> None:
    """Fail at import time when selectable configs reference missing runtimes."""
    identifier = re.compile(r"^[A-Za-z0-9_.-]+$")
    invalid = [
        name
        for name in {*RUNTIME_CONFIGS, *CODE_CONFIGS}
        if not identifier.fullmatch(name)
    ]
    if invalid:
        msg = f"config identifiers must be artifact-safe: {invalid}"
        raise RuntimeError(msg)
    for name, config in CODE_CONFIGS.items():
        runtime_config(config["runtime"])
        conversation = config["conversation_runtime"]
        if conversation is not None:
            runtime_config(conversation)
        if name != config["runtime"]:
            msg = (
                "selectable config identifiers must currently match their code "
                f"runtime identifiers: {name!r} != {config['runtime']!r}"
            )
            raise RuntimeError(msg)
    runtime_config(DEFAULT_CONVERSATION_RUNTIME)


validate_registry()


def main(argv: list[str] | None = None) -> int:
    """Print workflow-safe metadata for a registered runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime")
    args = parser.parse_args(argv)
    config = runtime_config(args.runtime)
    # One value per line lets Bash consume the result without evaluating it.
    print(config["graph"])
    print(config["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
