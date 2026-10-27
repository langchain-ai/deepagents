---
type: model and profile configuration
title: dcode Models and Profiles
description: Explains SDK provider and harness profiles alongside dcode model configuration, discovery, selection, runtime switching, capability profiles, reasoning effort, retries, and evaluation-specific model construction.
tags: [models, sdk-profiles, dcode, provider-configuration, model-catalog, reasoning]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-27T08:05:40.296Z
sources:
  - id: openwiki-source-aaf94db4700b8db4f60bdba9
    resource: repo://libs/code/deepagents_code/_glm_5p2_profile.py
  - id: openwiki-source-eea3092c31d2024ecf35cabd
    resource: repo://libs/code/deepagents_code/_paths.py
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-55d5c39401ac52584ce1f973
    resource: repo://libs/code/deepagents_code/configurable_model.py
  - id: openwiki-source-5afb93f98c34330c27609db2
    resource: repo://libs/code/deepagents_code/model_catalog.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-c101168dc0286ff6c29ed37f
    resource: repo://libs/code/deepagents_code/model_retry.py
  - id: openwiki-source-bdf7871023d068e30942c1ba
    resource: repo://libs/code/deepagents_code/reasoning_effort.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-a639ca6fd921316ba61b2a2a
    resource: repo://libs/code/tests/unit_tests/test_glm_5p2_profile.py
  - id: openwiki-source-e012c5898b6bc6cb1317467d
    resource: repo://libs/code/tests/unit_tests/test_model_catalog.py
  - id: openwiki-source-0dfd43fc1de1af401946bc38
    resource: repo://libs/code/tests/unit_tests/test_model_config.py
  - id: openwiki-source-50173942904153d619b9ae0d
    resource: repo://libs/deepagents/deepagents/_models.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-b27554b5c0e5b26fae2efb38
    resource: repo://libs/deepagents/deepagents/profiles/__init__.py
  - id: openwiki-source-f94d6bc3bb6ebd1565c1732f
    resource: repo://libs/deepagents/deepagents/profiles/_builtin_profiles.py
  - id: openwiki-source-59612eea63cbfafbd628feda
    resource: repo://libs/deepagents/deepagents/profiles/harness/harness_profiles.py
  - id: openwiki-source-1098130d42873f13aba9f5c2
    resource: repo://libs/deepagents/deepagents/profiles/provider/provider_profiles.py
  - id: openwiki-source-38c26264bf4a1dadb3401e80
    resource: repo://libs/deepagents/tests/unit_tests/test_models.py
  - id: openwiki-source-dd120a1be03e34bad3c59b22
    resource: repo://libs/evals/deepagents_harbor/langgraph_project/langgraph_agent.py
generated: { by: "openwiki/0.4.2", at: "2026-10-27T08:05:40.296Z" }
---

# dcode Models and Profiles

“Profile” has several deliberately separate meanings. The Deep Agents SDK uses **provider profiles** to construct a model from a string and **harness profiles** to adapt the agent graph. dcode uses **capability profiles** to describe a model for selection, prompt metadata, and reasoning controls; it also has a launch-selected user profile that owns configuration and state. Installing an integration or registering an SDK profile does not, by itself, make a model selectable, credentialed, or permitted in dcode.

## The profile boundaries

| Kind | Owner and lookup key | Responsibility |
| --- | --- | --- |
| User profile | dcode launch-time `DEEPAGENTS_HOME` | Trust root containing `config.toml`, credentials/state, and user agent material. |
| SDK `ProviderProfile` | `provider` or `provider:model` | Constructor kwargs, dynamic kwargs, and pre-initialization work for a **string** model spec. |
| SDK `HarnessProfile` | `provider` or `provider:model` | Prompt, tool, middleware, and default-subagent changes while assembling a Deep Agents graph. |
| dcode capability profile | `provider:model` | Model capabilities such as context window, modalities, tool calling, and reasoning levels. |

`DEEPAGENTS_HOME` is selected from the inherited launch environment and captured in the process-wide `PATHS` snapshot; project/global dotenv files cannot change it. This makes the selected user profile a trust boundary, rather than a setting that can drift after a directory change or reload. Its `config.toml` is the default source for `ModelConfig`; managed policy is included for default-path reads.

```mermaid
flowchart TD
    Input["Model string or BaseChatModel"] --> Resolve["SDK resolve_model"]
    Resolve --> IsString{"String specification"}
    IsString -->|Yes| Provider["ProviderProfile then init_chat_model"]
    IsString -->|No| Model["Existing BaseChatModel"]
    Provider --> Model
    Model --> Harness["HarnessProfile during graph assembly"]
    Harness --> Agent["Compiled Deep Agents graph"]
    DcodeConfig["dcode config and capability profile"] --> DcodeBuild["dcode create_model"]
    DcodeBuild --> Model
```

The SDK construction/assembly path is distinct from dcode's policy-gated construction path; capability data informs dcode behavior around the resulting model.

## SDK provider and harness profiles

The beta SDK registries are additive. Registrations accept a provider or exact `provider:model` key, and a later registration layers fields rather than replacing the old registration. `resolve_model()` leaves an existing `BaseChatModel` unchanged; only a string gets `ProviderProfile` handling and `init_chat_model()`. A supplied instance can still receive a harness profile if its provider and identifier can be determined.

Provider-profile resolution overlays an exact model registration on the provider registration. Static kwargs merge base first, hooks and dynamic factories run base first, factory output overrides static kwargs, and explicit caller kwargs win last. A hook or factory failure stops construction. Bootstrap is lazy, thread-coordinated, and once per process: SDK built-ins must load successfully, whereas faulty third-party entry points are logged and skipped.

Harness lookup similarly prefers an exact spec and falls back to provider-level data. `create_deep_agent()` resolves it separately for the main model and every declarative subagent before applying prompt, tool-description, and middleware changes. Runtime-only extra middleware can affect SDK-assembled main, general-purpose, and synchronous declarative subagent stacks; it cannot alter an already compiled or remote async subagent, so file-friendly `HarnessProfileConfig` excludes that extension.

Harness exclusions are not authorization controls. Required filesystem and subagent scaffolding cannot be removed, unmatched exclusions fail assembly, and tool exclusion runs after tool-injecting and custom middleware. This ordering makes tool exclusion the supported way to hide an injected tool from the model.

## dcode configuration, packages, and capability data

The distribution pins `deepagents==0.7.32`. Anthropic, Google GenAI, and OpenAI integrations are core dependencies; other supported integrations are optional provider extras, including the `all-providers` convenience extra. Packaging only makes an integration importable: enablement, credentials, configured lists, capability requirements, and `models.allowed` separately control usability.

A provider configuration can supply `enabled`, explicit `models`, credential/base-URL names, constructor `params`, capability `profile` overrides, or a trusted `class_path`. The latter imports and instantiates arbitrary Python from user configuration and consequently has the same trust implications as other local executable configuration. Provider-wide `params` and `profile` fields can contain per-model subtables; the per-model mapping shallowly wins.

`get_model_profiles()` reads upstream integration `_profiles` data, overlays configured capability fields, then applies a CLI `--profile-override` when supplied. Entries retain which keys were overridden. It also supports configured-only models, class-path packages exposing `_profiles`, mirrored Codex capabilities, and discovered Ollama metadata. Capability profiles are descriptive data, not SDK `ProviderProfile` constructor behavior.

### Catalog, discovery, and policy

`load_model_catalog()` returns a strict presentation snapshot for the inference host. It carries displayable scalar capabilities, override provenance, readiness, policy metadata, model specifications, and an optional current specification—not arbitrary constructor kwargs or profile extensions. A remote picker uses the supplied catalog rather than inspecting its local provider state and must apply catalog policy before installation or authentication prompts.

Discovery includes enabled providers whose upstream profile advertises tool calling and usable text input/output, then merges explicit configured models. It can derive profile data from a `class_path` package and query installed Ollama models. `models.allowed` filters the final lineup; current and recommended specifications remain eligible only when enablement and policy allow. Readiness remains independent of rows, so a known provider can offer setup guidance even when it has no discovered models.

`models.allowed` is a security boundary: absent means unrestricted, an empty tuple denies every model, and non-empty entries are exact specs or `provider:*` wildcards. A malformed declared allowlist fails closed as deny-all instead of becoming unrestricted.

```mermaid
flowchart TD
    Requested["Requested model specification"] --> Infer["Parse or infer provider"]
    Infer --> Allowed{"Allowed by models.allowed"}
    Allowed -->|No| Denied["ModelNotAllowedError"]
    Allowed -->|Yes| Credential["Bridge credentials"]
    Credential --> Defaults["Config defaults and SDK provider profile"]
    Defaults --> Runtime["Runtime kwargs"]
    Runtime --> Build["Custom class Codex or init_chat_model"]
    Build --> Metadata["Capabilities and retry budget"]
```

dcode checks the allowlist after canonical provider inference but before credential bridging, SDK profile work, or provider imports. Afterward it layers provider and per-model configuration, credentials, applicable SDK provider-profile defaults, then runtime `extra_kwargs` last. Codex uses its OAuth-aware path; configured custom classes bypass ordinary `init_chat_model()`. dcode applies configured and CLI capability-profile overrides to the concrete model; a missing profile warns, but an explicit `tool_calling=False` is fatal for an agent.

## Runtime selection, reasoning effort, and retries

`ConfigurableModelMiddleware` reads `model` and `model_params` from LangGraph runtime context on each call. It creates a replacement only when the requested specification differs, shallow-merges call settings, strips Anthropic-only settings when moving to another provider, and patches the Model Identity prompt section. A blocked switch always propagates. Other construction errors propagate in strict mode or log and retain the current model otherwise; the fallback deliberately does not overwrite checkpointed parameters with unknown values. Successful model calls persist the resolved spec and runtime settings needed for resume, while cache identity records effective configured settings separately. Async paths offload blocking construction and configuration reads.

Reasoning effort is driven by dcode capability profiles, not harness profiles. It is available only where `reasoning_output=True` and valid levels are advertised. dcode recognizes canonical and provider-native representations, removes old effort fields while preserving unrelated parameters, and stores the portable `reasoning_effort` value. It persists a selected effort by model in the active configuration; an explicit session/native setting takes precedence over restoring that preference. Anthropic `thinking.type="between_tools"` limits the offered levels to `low`, `medium`, and `high`.

Construction resolves a provider/CLI retry budget, disables recognized provider SDK retries, and stamps that budget on the concrete model. `CodeModelRetryMiddleware` uses the per-request model budget at the model node, so a runtime switch does not multiply retries or replay completed tools. It retries only classified transient failures, honoring valid capped `Retry-After` or jittered exponential backoff within the interactive delay budget; it re-raises non-transient or exhausted errors and emits attempt/retry lifecycle events for streamed-output reconciliation.

## GLM-5.2 dcode harness and Harbor evaluation construction

For three exact GLM-5.2 specs—Fireworks, OpenRouter, and Baseten—dcode registers a prompt-only `HarnessProfile` before creating its CLI graph. Registration is process-idempotent and deliberately does not replace an existing suffix for the same exact spec. This is dcode-specific harness adaptation, separate from both provider construction profiles and capability data.

Headless dcode main and generated subagent stacks also install a Fireworks-only terminal-stall recovery middleware. If exactly `fireworks:accounts/fireworks/models/glm-5p2` returns one tool-free AI result with `finish_reason == "length"`, it makes one recovery call with a trusted suffix, `tool_choice="any"`, and nested Fireworks `reasoning_effort="none"`. It never loops, does not run in interactive sessions, and does not apply to the same model identity on OpenRouter or Baseten.

Harbor builds eval models directly with `init_chat_model()`, so it bypasses Deep Agents provider-profile construction. Its eval adapter explicitly sets `use_responses_api=True` for `openai:` specs when the caller did not set it. For the shared exact GLM-5.2 set, it defaults nested `model_kwargs.reasoning_effort` to `high` only when neither top-level nor nested native effort field was supplied. The adapter then creates either the dcode headless CLI graph or a bare SDK graph; for the CLI path it mirrors resolved model identity into dcode runtime state so the production Model Identity prompt is accurate.

## Safe changes and focused tests

Changing a provider is a multi-boundary change: choose package/extra, credential and endpoint mapping, discovery, policy, constructor behavior, capability data, reasoning representation, retry ownership, and catalog projection independently. Preserve the allowlist gate's position before side effects. Test capability-profile fallback and override provenance separately from SDK profile precedence.

Focused coverage includes profile discovery for configured/class-path/Ollama sources, deny-all and wildcard policy semantics, stored credential precedence, capability profile merging, native reasoning normalization, runtime switch fallback and checkpoint state, catalog redaction, and retry behavior. `test_glm_5p2_profile.py` verifies the narrow terminal-stall predicate, forced recovery request, non-mutation of the original request, and one-retry ceiling. Harbor construction should be tested separately because it intentionally bypasses SDK provider profiles.

## Related pages

- [Code agent](/openwiki/architecture/code-agent.md)
- [Configuration layering](/openwiki/concepts/config-layering.md)
- [Development](/openwiki/operations/development.md)
- [Run evaluations](/openwiki/workflows/run-evals.md)
