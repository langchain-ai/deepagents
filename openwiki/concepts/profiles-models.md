---
type: model and profile configuration
title: Profiles and Models
description: Explains SDK profiles and the separate dcode and Talon model control planes, including startup defaults, policy, catalog selection, retries, compaction, and classifier-backed Auto mode.
tags: [models, sdk-profiles, dcode, talon, provider-configuration, retries]
sources:
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-18abc7e59899514f067032b2
    resource: repo://libs/code/deepagents_code/auto_mode.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-2fb89d2b59c886d0cb3ee3ea
    resource: repo://libs/code/deepagents_code/config_manifest.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-52d96f61bc4737f02a18cf79
    resource: repo://libs/code/deepagents_code/configuration/resolver.py
  - id: openwiki-source-2d5bc2a1a6685429db181b39
    resource: repo://libs/code/deepagents_code/configuration/service.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-7e241f30f5c7753642ea34d5
    resource: repo://libs/code/deepagents_code/model_api.py
  - id: openwiki-source-5afb93f98c34330c27609db2
    resource: repo://libs/code/deepagents_code/model_catalog.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-ece74ecc87a2bc395b3ce368
    resource: repo://libs/code/deepagents_code/model_metadata.py
  - id: openwiki-source-c101168dc0286ff6c29ed37f
    resource: repo://libs/code/deepagents_code/model_retry.py
  - id: openwiki-source-c100a7d2ff8c43af8ad1b816
    resource: repo://libs/code/deepagents_code/offload_middleware.py
  - id: openwiki-source-e9a7ab29b270179f4eea7107
    resource: repo://libs/code/deepagents_code/tui/widgets/model_selector.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-e012c5898b6bc6cb1317467d
    resource: repo://libs/code/tests/unit_tests/test_model_catalog.py
  - id: openwiki-source-50173942904153d619b9ae0d
    resource: repo://libs/deepagents/deepagents/_models.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-b27554b5c0e5b26fae2efb38
    resource: repo://libs/deepagents/deepagents/profiles/__init__.py
  - id: openwiki-source-f94d6bc3bb6ebd1565c1732f
    resource: repo://libs/deepagents/deepagents/profiles/_builtin_profiles.py
  - id: openwiki-source-59612eea63cbfafbd628feda
    resource: repo://libs/deepagents/deepagents/profiles/harness/harness_profiles.py
  - id: openwiki-source-1098130d42873f13aba9f5c2
    resource: repo://libs/deepagents/deepagents/profiles/provider/provider_profiles.py
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-f04ce33d1db21a61b1e6e8b3
    resource: repo://libs/talon/deepagents_talon/model_selection.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
verified:
  - by: openwiki/0.4.2
    at: 2026-11-18T08:03:59.609Z
generated: { by: "openwiki/0.4.2", at: "2026-11-18T08:03:59.609Z" }
---

# Profiles and Models

"Profile" has different meanings at three boundaries. In the Deep Agents SDK, a **ProviderProfile** changes how a string model specification becomes a client and a **HarnessProfile** changes the agent graph around that client. In dcode, capability profiles are display and validation metadata, while `models.allowed` is the construction policy. Talon has its own environment-based catalog and turn-local switching; it is not a dcode session.

## Control-plane map

| Plane | Owner and key | Controls | Does not control |
| --- | --- | --- | --- |
| SDK `ProviderProfile` | Deep Agents; `provider` or `provider:model` | `init_chat_model` kwargs, dynamic kwargs, and pre-initialization checks | Prompts, tools, graph middleware |
| SDK `HarnessProfile` | Deep Agents; `provider` or `provider:model` | Prompts, tool visibility, middleware, and default subagent behavior | Model-client construction |
| dcode capability profile | dcode; `provider:model` | Advertised input, context, tool, and output capabilities | Client kwargs or admission |
| dcode `models.allowed` | dcode configuration | Whether dcode may construct a canonical model | Provider enablement or display metadata |
| Talon catalog | Talon host environment | Models an interactive Talon chat may select | dcode policy, auth store, and retries |

## SDK profiles: construction versus harness

Both SDK registries are beta APIs. Registration layers settings onto an existing registration rather than replacing it, so applications should register before resolving models or building graphs.

```mermaid
flowchart TD
    Input["String specification or model instance"] --> Existing{"Already a model instance"}
    Existing -->|Yes| Kept["Keep construction unchanged"]
    Existing -->|No| Provider["Resolve ProviderProfile"]
    Provider --> Client["Construct with init_chat_model"]
    Kept --> Harness["Resolve HarnessProfile"]
    Client --> Harness
    Harness --> Agent["Assemble Deep Agents graph"]
```

*Provider profiles participate in string-model construction, while harness profiles shape the assembled agent.*

### ProviderProfile

`resolve_model()` passes an existing `BaseChatModel` through unchanged. For a string, it applies the resolved provider profile and then calls `init_chat_model`. A provider-wide profile supplies the base; an exact `provider:model` profile overlays it. Static values merge with the exact value winning, `pre_init` hooks and factories run base first, factory results override static values, and explicit caller kwargs have final precedence. A failed pre-initialization hook prevents its factory and client construction.

Use ProviderProfile for constructor defaults, environment-derived settings, attribution headers, and preflight checks—not graph behavior. Its registered `init_kwargs` are defensively copied and immutable, preventing later mutation of the caller's mapping from changing model construction.

### HarnessProfile and assembly limits

`create_deep_agent()` resolves the main model's harness profile and resolves profiles for declarative synchronous subagents from each subagent's model. Exact model fields overlay provider fields before prompt, tool-description, and middleware changes. Prompt fragments assemble in `USER`, `BASE`, `SUFFIX` order.

A harness profile can change tool descriptions, hide model-visible tools, alter the default general-purpose subagent, and add static or factory `extra_middleware`. It reaches the main agent, generated general-purpose subagent, and SDK-assembled declarative synchronous subagents. It cannot modify an already compiled subagent or remote asynchronous subagent, whose graphs are independently owned. The file-friendly `HarnessProfileConfig` deliberately excludes runtime-only middleware.

Middleware exclusions cannot remove required filesystem or synchronous-subagent scaffolding, and an unmatched exclusion fails assembly. Prefer `excluded_tools` when the aim is to hide a tool from the model: this filter runs after tool-injecting and custom middleware. It is a model-visibility control, not authorization.

Profile bootstrap is lazy, coordinated across threads, and runs once. Built-ins load directly before third-party provider and harness entry-point hooks layer on top. A broken built-in fails bootstrap and restores the prior registry; a broken third-party plugin is logged and skipped.

`create_deep_agent()` adds prompt-caching middleware to main, generated general-purpose, and declarative synchronous-subagent stacks. Anthropic caching activates lazily per Anthropic request. Bedrock and Fireworks are optional integrations: an absent module omits the integration, while unrelated import failures propagate. The lookup caches classes or their absence but constructs a fresh middleware instance for every stack.

## dcode: startup configuration and construction

The manifest defines model defaults: `[models].default` is the launch default, `recent` records a switch, `summarization_default` supplies an optional compaction model, and `auto_classifier` supplies an optional Auto reviewer. `auto_classifier_timeout` is a 1–300-second decision-batch limit. `models.providers` may declare custom providers, but its `class_path` is a trusted code boundary: dcode imports and instantiates that `BaseChatModel` class with process privileges.

Manifest resolution ranks managed policy, CLI, retained runtime reload, environment, user `config.toml`, and typed defaults. Managed configuration outranks user configuration; `models.allowed` is an enforced managed-policy key.

```mermaid
flowchart TD
    Request["Requested model specification"] --> Resolve["Canonicalize provider and model"]
    Resolve --> Allowed{"models.allowed permits it"}
    Allowed -->|No| Denied["Raise ModelNotAllowedError"]
    Allowed -->|Yes| Credentials["Bridge and check credentials"]
    Credentials --> Profile["Layer configuration and ProviderProfile"]
    Profile --> Retries["Resolve retry budget and disable known SDK retries"]
    Retries --> Build["Codex class_path or init_chat_model"]
    Build --> Result["Return ModelResult"]
```

*dcode performs policy admission before credential work, provider hooks, or provider imports.*

`create_model()` accepts `provider:model` or a bare name, infers and canonicalizes the provider, and returns a `ModelResult` rather than immediately mutating global runtime state. The result carries resolved identity, profile-derived metadata, the effective retry budget, and the explicit CLI retry override needed for later cross-provider switching.

The allowlist runs after inference but before credential bridging, ProviderProfile hooks or factories, and provider imports. A denied model therefore causes none of those side effects. After admission, dcode may bridge stored `/auth` credentials into the environment and performs an early credential check except for implicit-auth providers; errors are typed and actionable.

Construction layers provider and per-model configuration, stored credentials, and SDK ProviderProfile output before `extra_kwargs`, which normally win. dcode converts arbitrary ProviderProfile hook/factory errors into `ModelConfigError`. It constructs `openai_codex` through its OAuth-aware model class, a configured `class_path` directly, or other providers through `init_chat_model`. Afterwards it overlays configured capability values and then `--profile-override` values; an assignment rejected by the model fails strictly. Missing capability data warns, but explicit `tool_calling=False` fails agent-model validation.

### Retries and compaction

dcode owns the visible retry budget in model-node middleware. Its precedence is `--max-retries`, `[retries.<provider>].max_retries`, `[retries].max_retries`, then the built-in default of `5`; `0` disables retries. Known provider SDK retry loops are forced off after kwargs merge so retries are not multiplied. The concrete model receives the effective budget, allowing request-time model switches to use their own provider policy rather than stale middleware state. Auxiliary callers such as summary generation use the stamped budget too, with a default-and-warning fallback for unstamped models.

`--summarization-model` overrides `[models].summarization_default`; an absent or blank override reuses the main model. Dedicated construction is deferred until compaction, avoiding provider imports on the pre-first-paint path. `_LazySummaryModel` serializes setup and invocation, warns once and falls back to the main model if construction fails, and restores main-model token-counting, trim, and retry tuning if a dedicated invocation fails. A successful dedicated model gets dcode retry handling, provider-specific counting, and a history budget of four fifths of a positive input window; this preserves `--max-retries 0` rather than using LangChain's generic summary retries.

## Classifier-backed Auto mode

Auto is an authorization path, not a general model-selection convenience. `AutoModeHITLMiddleware` first applies deterministic policy and sends the remaining tool-call batch to a structured-output classifier. The classifier defaults to the main agent model when no separate model is selected. A distinct `provider:model` comes from `--auto-classifier-model`, `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL`, or `[models].auto_classifier`; when none is configured, startup may select a lightweight default for the main provider. An explicitly blank CLI flag is preserved as an inherit sentinel, overriding configured/default classifiers and returning review to the main model.

```mermaid
flowchart TD
    Calls["Gated tool-call batch"] --> Policy["Deterministic Auto policy"]
    Policy --> Fixed{"Decision already fixed"}
    Fixed -->|Yes| Route["Apply policy decision"]
    Fixed -->|No| Select["Resolve per-run or startup classifier"]
    Select --> Review["Structured classifier review"]
    Review --> Valid{"Valid response"}
    Valid -->|Yes| Route
    Valid -->|No| Deny["Deny unavailable batch or require human"]
    Deny --> Route
```

*Auto only authorizes a batch after deterministic policy or a validated classifier result; classifier failures do not silently approve it.*

A runtime `classifier_model` context value wins over startup configuration, enabling `/auto model` without rebuilding the graph. A distinct classifier is built lazily in the workspace environment and cached by specification; construction has a separate timeout, and review has the configured batch deadline. The distinct construction disables preserved-thinking binding because classifier history does not replay thinking blocks. A failure to build or invoke a classifier fails closed: the affected batch is denied as unavailable or escalated to human approval after the configured fallback conditions. A repeated construction fault is latched until a successful review, so later batches ask for human approval instead of silently repeating denials. Counter-state write failures also require human approval.

Startup performs cheap policy, provider-recognition, and credential checks for a configured classifier without importing LangChain. A policy-blocked classifier is downgraded to inheriting the already-approved main model so it cannot abort agent construction; other detected problems are warned and remain fail-closed at review time. The classifier uses the same `create_model()` policy and retry ownership when it is actually built.

## dcode inference-host catalog and picker

In a client/server session, the workspace-bound inference host—not the client's Python environment—owns catalog discovery, provider readiness, canonical model resolution, and display metadata. The host returns strict presentation-only payloads. `ModelCatalog` includes model specifications, scalar capability fields, provider readiness/labels, policy presentation, and the retained current spec; constructor kwargs and arbitrary extensions do not cross the boundary. `ModelMetadata` is a frozen, validated, provider-independent exchange type and never contains a model object or credentials.

```mermaid
sequenceDiagram
    participant Picker as TUI picker
    participant Client as Remote client
    participant Host as Inference host
    participant Workspace as Bound workspace
    Picker->>Client: Load catalog and current specification
    Client->>Host: Request thread catalog
    Host->>Workspace: Resolve configuration and environment
    Host->>Host: Discover allowed models and readiness
    Host-->>Client: Presentation-only catalog
    Client-->>Picker: Validated ModelCatalog
    Picker->>Client: Validate proposed model
    Client->>Host: Resolve without inference
    Host-->>Picker: Canonical ModelMetadata
```

*The inference host owns discovery and validation; the client renders and commits only safe metadata.*

Discovery applies enabled-provider and allowlist policy, surfaces allowed recommendations, retains an allowed active spec if discovery omits it, and separately includes known enabled providers so setup/readiness can be presented without model rows. Per-provider discovery failures are logged and skipped. Main-role catalog requests use main `profile_overrides`; auxiliary roles do not. The recommended-only selector also retains the active model and recent selections while catalog loading preserves typed filter input.

Before persisting a main, classifier, or summarization default, the picker resolves the selection asynchronously through the inference host and writes only the returned canonical specification on success. It avoids duplicate work and does not save after dismissal or failure. Explicit catalog models rejected by `models.allowed` are stopped before installation or authentication setup; unavailable but allowed providers can lead to setup flows. Ambiguous model strings are deferred to host resolution rather than guessed locally.

## Talon is a separate runtime

Talon does not use dcode's `models.allowed`, `/auth` credential store, `ModelResult`, or retry policy. It applies SDK `apply_provider_profile` before `init_chat_model`, then calls `create_deep_agent`; compatible SDK provider and harness profiles and prompt caching therefore apply to its main agent and SDK-assembled synchronous subagents.

Talon discovers tool-capable models advertised by installed provider packages and limits them to credentials in Talon's own environment. It accepts an interactive switch only when the exact specification is in that catalog (the graph default is always included), builds and caches the accepted model before activation, and binds it in turn-local state. `ModelSelectionMiddleware` substitutes it for main-agent calls, while Talon replaces the summarizer slot so context sizing follows the selected model, all without graph recompilation. Failed or stale selections leave the default model usable; subagents retain their own models.

## Compatibility and change checklist

`deepagents-code` 0.2.10 requires Python `>=3.14,<3.15` and pins `deepagents==0.7.39`. Deep Agents 0.7.39 supports Python `>=3.11,<4.0` and requires LangChain `>=1.5.0,<2.0.0`, LangChain Core `>=1.8.1,<2.0.0`, `langchain-anthropic >=1.7.10,<2.0.0`, and `langchain-google-genai >=4.4.3,<5.0.0`. dcode includes Anthropic, Google GenAI, and OpenAI integration floors, with further provider integrations as extras. `deepagents-talon` 0.1.0 requires Python `>=3.14,<3.15` and depends on `deepagents>=0.7.39` and `deepagents-code>=0.2.9,<1.0.0`.

Focused changes should preserve these invariants: admission precedes construction side effects; only the host determines remote readiness; catalog payloads remain presentation-safe; model defaults are validated before persistence; dcode has one retry owner; an unavailable Auto classifier never grants approval; and Talon's catalog/environment does not become a back door into dcode policy.

## Related pages

- [Code agent architecture](/openwiki/architecture/code-agent.md)
- [Configuration layering](/openwiki/concepts/config-layering.md)
- [ACP integration](/openwiki/integrations/acp.md)
- [Development](/openwiki/operations/development.md)
- [Run a dcode session](/openwiki/workflows/run-dcode-session.md)
