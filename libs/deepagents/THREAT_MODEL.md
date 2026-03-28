# Threat Model: deepagents (SDK library)

> Generated: 2026-03-28 | Commit: e859077f | Scope: `libs/deepagents/deepagents/` — SDK library only (not CLI)

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference — findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. We welcome suggestions and corrections to improve this document.

## Scope

### In Scope

- `deepagents/graph.py` — primary public API (`create_deep_agent`, `resolve_model`, `get_default_model`)
- `deepagents/_models.py` — model resolution helpers (`resolve_model`, `check_openrouter_version`, `get_model_identifier`)
- `deepagents/backends/` — all backend implementations (`StateBackend`, `FilesystemBackend`, `LocalShellBackend`, `StoreBackend`, `CompositeBackend`, `BaseSandbox`, `LangSmithSandbox`) and the `BackendProtocol` / `SandboxBackendProtocol` protocol definitions
- `deepagents/middleware/` — all middleware: `FilesystemMiddleware`, `MemoryMiddleware`, `SkillsMiddleware`, `SubAgentMiddleware`, `AsyncSubAgentMiddleware`, `SummarizationMiddleware`, `PatchToolCallsMiddleware`

### Out of Scope

- `libs/cli/` — CLI product (separate threat model; CLI and SDK are treated as independent product stories)
- `libs/acp/`, `libs/evals/`, `libs/partners/`, `libs/harbor/` — separate packages
- `examples/`, `tests/`, `scripts/` — not shipped code; tests are read for context only
- User application code, prompt construction, model selection, and model behavior — user-controlled
- Deployment infrastructure (hosting, auth, network access controls) — user/deployer-controlled
- LangGraph internals beyond what is directly invoked by this library
- Model provider APIs (Anthropic, OpenAI, Google, OpenRouter) — external
- Remote LangGraph server deployments used by `AsyncSubAgentMiddleware` — user/deployer-controlled

### Assumptions

1. The project is used as a Python library — users control their own application code, model selection, deployment topology, and backend configuration.
2. `StateBackend` (default) stores files in ephemeral LangGraph agent state; it does not persist data across threads unless the user also provides a `checkpointer`.
3. `LocalShellBackend` is **not the default**; it must be explicitly provided by the user. It carries documented security warnings.
4. `FilesystemBackend` without `virtual_mode=True` provides no path restriction; this is documented and expected for local dev use cases. The default (`virtual_mode=None`) is deprecated and will change to `True` in v0.5.0.
5. Users who require isolation for untrusted workloads are expected to extend `BaseSandbox` or use container/VM-level sandboxing — the library does not provide OS-level process isolation.
6. Memory and skill files are user-controlled artifacts loaded from user-specified paths; the library does not provide or vouch for their content.
7. `AsyncSubAgentMiddleware` connects to user-configured remote LangGraph server URLs. Authentication relies on environment variables (`LANGGRAPH_API_KEY` / `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY`) read by the LangGraph SDK — the library does not manage authentication credentials.
8. When the `openai:` model prefix is used via `resolve_model`, the OpenAI Responses API is used by default, which retains conversation data on OpenAI servers unless the user explicitly opts out.

---

## System Overview

`deepagents` is an open source Python SDK built on LangGraph and LangChain for creating "deep agents" — LLM-powered agents with sub-agent spawning, filesystem operations, todo lists, memory, and skills. The library ships a factory function (`create_deep_agent`) that assembles a LangGraph `CompiledStateGraph` from user-provided tools, models, middleware, and storage backends. It is intended for use in developer-facing CLI tools, research pipelines, and application backends where users control the execution environment. It does **not** run a server itself — it compiles a graph that user application code invokes.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       User Application                              │
│                                                                     │
│  create_deep_agent(model, tools, backend, checkpointer, store,      │
│                    subagents=[AsyncSubAgent(...), ...])              │
│          │                                                          │
│          ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │         Middleware Stack (framework-controlled)              │    │
│  │  TodoList → Memory → Skills → Filesystem → SubAgent →       │    │
│  │  AsyncSubAgent → Summarization → PromptCaching →            │    │
│  │  PatchToolCalls                                              │    │
│  └─────────────────────────┬──────────────────────────────┬────┘    │
│                            │                              │         │
│     ┌──────────────────────┼──────────────┐              │         │
│     ▼                      ▼              ▼              ▼         │
│  ┌────────┐        ┌────────────┐  ┌────────────┐  ┌──────────┐    │
│  │  LLM   │        │  Backend   │  │ Inline     │  │ Remote   │    │
│  │(external│       │ (user cfg) │  │ Subagent   │  │ LangGraph│    │
│  └────────┘        └─────┬──────┘  │ (same stk) │  │ Server   │    │
│                          │         └────────────┘  │(external)│    │
│  - - - - - - - - - - - - │- - - - - - - - - - - -  └──────────┘    │
│  Trust Boundary:          ▼       Backends cross here              │
│              ┌──────────────────────┐                              │
│              │ State / Store / FS / │                              │
│              │ LocalShell / Sandbox │                              │
│              └──────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Classification

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | Conversation history | Messages, tool calls, tool results in LangGraph state | High | LangGraph state; optional checkpointer (user-controlled) | Depends on user's checkpointer backend | Per-thread; persists if checkpointer provided | GDPR if personal data discussed |
| DC2 | Memory file contents | AGENTS.md body text | High | User-controlled backend paths; injected into system prompt | Depends on backend; `StateBackend` holds in-process | Indefinite until user removes files | None direct |
| DC3 | Skill file contents | SKILL.md YAML + markdown | Medium | User-controlled backend paths | Depends on backend | Indefinite until user removes files | None direct |
| DC4 | Shell command output | stdout/stderr from LocalShellBackend | Critical | Transient in-process string; truncated at 100KB; not persisted by framework | Single request lifetime | None (in-process only) | All if contains secrets |
| DC5 | LLM API credentials | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc. | Critical | Environment variables only; framework reads via LangChain providers | None (env var) | Process lifetime | All — breach trigger |
| DC6 | OpenAI Responses API retained data | Full conversation data sent via OpenAI Responses API | High | OpenAI servers (external) | OpenAI-controlled | Per OpenAI retention policy unless `store=False` | GDPR, data sovereignty |
| DC7 | Async task state | `thread_id`, `run_id`, task status, run output | Medium | `async_tasks` key in LangGraph agent state | Depends on user's checkpointer backend | Per-thread; persists if checkpointer provided | None direct |

### Data Classification Details

#### DC1: Conversation History

- **Fields**: Full `HumanMessage`, `AIMessage`, `ToolMessage` sequence in LangGraph state.
- **Storage**: Ephemeral in `StateBackend` (default); persisted if user provides a `checkpointer`.
- **Access**: In-process only for `StateBackend`; depends on checkpointer backend for persistent storage.
- **Encryption**: None by default; depends on user's checkpointer backend.
- **Retention**: Per-thread for `StateBackend`; unbounded for persistent checkpointers.
- **Logging exposure**: Tool call results (file contents, shell output, web content) are stored in the message history. If a checkpointer persists this, all tool output is persisted.
- **Gaps**: No retention policy enforced by the framework. Shell output (DC4) re-enters DC1 and may be persisted.

#### DC4: Shell Command Output

- **Fields**: Combined stdout/stderr from `LocalShellBackend.execute`.
- **Storage**: In-process string only; capped at 100KB (`local_shell.py:LocalShellBackend.__init__`, `max_output_bytes`).
- **Access**: Available in the agent's context window; forwarded to the LLM provider via DC1.
- **Encryption**: None.
- **Retention**: Single request lifetime in-process; persisted in DC1 if checkpointer is configured.
- **Logging exposure**: Shell commands may output `/etc/passwd`, SSH keys, environment variables, or other secrets. This output enters the LLM context window and may be sent to external LLM providers (DC5/DC6 implications).
- **Gaps**: No content filtering on shell output before it enters the agent context.

#### DC5: LLM API Credentials

- **Fields**: Provider API keys read from environment via LangChain providers.
- **Storage**: Environment variables only; framework does not write keys to disk or log them.
- **Access**: Available to the process and any subprocess spawned with `inherit_env=True`.
- **Encryption**: None.
- **Retention**: Process lifetime.
- **Logging exposure**: `MemoryMiddleware` system prompt explicitly instructs the agent "Never store API keys, access tokens, passwords..." (`memory.py:MEMORY_SYSTEM_PROMPT`). If a user's AGENTS.md contains credentials, they will be injected verbatim into LLM requests.
- **Gaps**: With `LocalShellBackend(inherit_env=True)`, all env vars (including API keys) are available to every shell command.

---

## Components

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | Agent Factory (`graph.py`) | Assembles and returns a `CompiledStateGraph`; the primary user-facing API | framework-controlled | Yes | `graph.py:create_deep_agent`, `graph.py:resolve_model`, `graph.py:get_default_model` |
| C2 | StateBackend | Stores files in LangGraph agent state (ephemeral, per-thread) | framework-controlled | **Yes** (default backend) | `backends/state.py:StateBackend.__init__`, `StateBackend.read`, `StateBackend.write`, `StateBackend.edit` |
| C3 | FilesystemBackend | Reads/writes files directly from the host filesystem | user-controlled | No (opt-in) | `backends/filesystem.py:FilesystemBackend.__init__`, `FilesystemBackend._resolve_path` |
| C4 | LocalShellBackend | FilesystemBackend + unrestricted local shell execution via `subprocess.run(shell=True)` | user-controlled | **No** (must be explicitly provided) | `backends/local_shell.py:LocalShellBackend.execute` |
| C5 | StoreBackend | Persistent cross-thread storage via LangGraph `BaseStore` | user-controlled | No (opt-in, requires `store=`) | `backends/store.py:StoreBackend.__init__`, `StoreBackend._validate_namespace` |
| C6 | CompositeBackend | Routes file operations by path prefix to multiple backends | user-controlled | No (opt-in) | `backends/composite.py:CompositeBackend.__init__` |
| C7 | MemoryMiddleware | Loads AGENTS.md files from backend and injects into system prompt | framework-controlled | No (opt-in, `memory=` param) | `middleware/memory.py:MemoryMiddleware.modify_request` |
| C8 | SkillsMiddleware | Loads SKILL.md files from backend, parses YAML, injects metadata into system prompt | framework-controlled | No (opt-in, `skills=` param) | `middleware/skills.py:SkillsMiddleware.modify_request`, `_parse_skill_metadata` |
| C9 | SubAgentMiddleware | Adds a `task` tool that spawns ephemeral inline sub-agents with inherited state | framework-controlled | **Yes** (always added by `create_deep_agent`) | `middleware/subagents.py:SubAgentMiddleware.__init__`, `_build_task_tool` |
| C10 | LLM Provider | External API (Anthropic, OpenAI, Google, OpenRouter, etc.) called with agent messages | external | Yes | HTTP via `langchain-anthropic`, `langchain-openai`, `langchain-google-genai` |
| C11 | LangGraph Checkpointer | Optional persistent state checkpointing via user-supplied `Checkpointer` | user-controlled | No (opt-in, `checkpointer=` param) | `graph.py:create_deep_agent` (passed through to `create_agent`) |
| C12 | AsyncSubAgentMiddleware | Manages background tasks on remote LangGraph server deployments via LangGraph SDK | framework-controlled | No (opt-in, added when `AsyncSubAgent` specs are provided) | `middleware/async_subagents.py:AsyncSubAgentMiddleware`, `_build_launch_tool`, `_build_check_tool` |
| C13 | BaseSandbox | Base class for sandboxed backends; implements file ops via `execute()` + base64-encoded parameters | framework-controlled | No (abstract) | `backends/sandbox.py:BaseSandbox.read`, `BaseSandbox.write`, `BaseSandbox.edit` |
| C14 | FilesystemMiddleware | Creates file/shell tools from backend; controls which tools are exposed to the agent | framework-controlled | **Yes** (always added) | `middleware/filesystem.py:FilesystemMiddleware._create_execute_tool` |

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User / Framework | Where user-provided values (model, tools, callbacks, config) enter framework code | Middleware stack, default tool implementations, agent graph structure | User's choice of model, tools registered, system prompt content, deployment environment |
| TB2 | Framework / LLM Provider | Where framework sends messages and receives LLM output | Message construction, tool call routing, retry behavior | Model behavior, what the model returns, model safety, OpenAI data retention policy |
| TB3 | Framework / Agent Code | Where LLM output (tool calls, responses) re-enters framework execution | Tool call dispatch; `SubAgentMiddleware` validates `subagent_type`; `AsyncSubAgentMiddleware` validates `subagent_type` | LLM-suggested tool names, arguments, and `description` content in task / async task calls |
| TB4 | Framework / Backend Storage | Where framework reads from and writes to storage backends | `BackendProtocol` interface; `StoreBackend._validate_namespace`; `BaseSandbox` base64 parameter encoding | Contents of files/state the user or LLM previously wrote; external storage system integrity |
| TB5 | Backend / Host OS | Where backend operations reach the host operating system | `FilesystemBackend._resolve_path` (virtual_mode=True only); `LocalShellBackend` timeout; output truncation | Host filesystem permissions, environment variables, installed tools, OS security controls |
| TB6 | Framework / Remote LangGraph API | Where `AsyncSubAgentMiddleware` makes outbound HTTP calls to remote LangGraph servers | LangGraph SDK client; URL and headers from user-provided `AsyncSubAgent` spec; auth from env vars | Remote server security, response content, availability, data handling |

### Boundary Details

#### TB1: User / Framework

- **Inside**: `create_deep_agent` validates model resolution (`_models.py:resolve_model`), constructs the middleware stack with framework defaults (TodoList, Filesystem, SubAgent, AsyncSubAgent, Summarization, PromptCaching, PatchToolCalls middleware), and assembles the LangGraph graph. Entry: `graph.py:create_deep_agent`.
- **Outside**: The user supplies `model`, `tools`, `system_prompt`, `backend`, `checkpointer`, `store`, `memory`, `skills`, `subagents`. The framework does not validate the safety or content of any of these.
- **Crossing mechanism**: Python function call arguments.

#### TB3: Framework / Agent Code (LLM Output Re-entry)

- **Inside**: LangGraph tool dispatch routes tool call names and arguments to registered tool implementations. `SubAgentMiddleware` validates `subagent_type` against the registered set (`middleware/subagents.py:_build_task_tool`). `_EXCLUDED_STATE_KEYS` prevents certain state keys from being passed to inline subagents. `AsyncSubAgentMiddleware` validates `subagent_type` against registered async subagent names at launch time.
- **Outside**: The content of `description` passed to `task` or `start_async_task` is LLM-generated and unvalidated. Tool arguments for user-registered tools are also LLM-generated.
- **Crossing mechanism**: LangGraph tool call dispatch; Python function call with LLM-provided arguments.

#### TB4: Framework / Backend Storage

- **Inside**: `StoreBackend._validate_namespace` rejects wildcard characters and enforces alphanumeric-plus-safe-chars constraint (`backends/store.py:_validate_namespace`, `_NAMESPACE_COMPONENT_RE`). `BaseSandbox` uses base64-encoded parameters to avoid shell escaping issues (`backends/sandbox.py:_GLOB_COMMAND_TEMPLATE`). `StateBackend` reads directly from `runtime.state["files"]` — no path validation.
- **Outside**: File content stored by the user or written by the agent in prior turns. The framework reads this content back without sanitization before injecting into system prompts (memory, skills).
- **Crossing mechanism**: `BackendProtocol.read`, `download_files`, `write` method calls.

#### TB5: Backend / Host OS

- **Inside** (virtual_mode=True only): `FilesystemBackend._resolve_path` checks `..` and `~` and verifies `full.relative_to(self.cwd)` (`backends/filesystem.py:FilesystemBackend._resolve_path`). `LocalShellBackend` runs commands with configurable timeout (`backends/local_shell.py:LocalShellBackend.execute`). Output capped at 100KB.
- **Outside**: Host filesystem layout, environment variables, OS user permissions, installed software. Shell commands with `shell=True` have full access regardless of `virtual_mode`.
- **Crossing mechanism**: `os.open` / `subprocess.run` with `shell=True` (LocalShellBackend).

#### TB6: Framework / Remote LangGraph API

- **Inside**: `AsyncSubAgentMiddleware` constructs a `langgraph_sdk` client using URL and headers from user-provided `AsyncSubAgent` spec. Auth credentials read from environment by the SDK.
- **Outside**: Remote server's response content. The framework does not validate response content before injecting it into conversation history.
- **Crossing mechanism**: HTTPS via `langgraph_sdk.get_client`.

---

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | C10 (LLM) | C9 (SubAgentMiddleware) | LLM-generated `task` description + `subagent_type` | DC1 | TB3 | function call (tool dispatch) |
| DF2 | C7 (MemoryMiddleware) | C10 (LLM) | AGENTS.md content → injected into system prompt | DC2 | TB4, TB2 | backend download → string append → API call |
| DF3 | C8 (SkillsMiddleware) | C10 (LLM) | SKILL.md YAML + markdown → injected into system prompt | DC3 | TB4, TB2 | backend download → YAML parse → string append → API call |
| DF4 | C1 (User) | C8 (SkillsMiddleware) | Skill source paths | — | TB1 | function argument |
| DF5 | C1 (User) | C7 (MemoryMiddleware) | Memory file paths | — | TB1 | function argument |
| DF6 | C1 (User) | C5 (StoreBackend) | Namespace factory callable | — | TB1 | function argument → callable invoked at runtime |
| DF7 | C10 (LLM) | C4 (LocalShellBackend) | LLM-generated shell command string | DC4 | TB3, TB5 | function call → `subprocess.run(shell=True)` |
| DF8 | C2/C3/C5 (Backend) | C9 (SubAgentMiddleware) | LangGraph state (files, todos, context) | DC1 | TB3 | state dict pass-through to subagent |
| DF9 | C11 (Checkpointer) | C10 (LLM) | Deserialized checkpoint state (messages, files) | DC1 | TB4, TB2 | LangGraph checkpoint load → agent state → prompt |
| DF10 | C12 (AsyncSubAgentMiddleware) | Remote LangGraph server | Task launch: `graph_id`, `description` (LLM-generated), `thread_id` | DC7 | TB6 | HTTPS via LangGraph SDK |
| DF11 | Remote LangGraph server | C12 (AsyncSubAgentMiddleware) | Task run output: status, results, errors | DC7 | TB6 | HTTPS via LangGraph SDK; injected into DC1 |
| DF12 | C4 (LocalShellBackend) | C10 (LLM) | Shell command output (stdout/stderr) re-entering context | DC4 | TB5, TB2 | ToolMessage → agent state → API call |

### Flow Details

#### DF2: MemoryMiddleware → LLM system prompt

- **Data**: Raw bytes from backend `download_files`, decoded UTF-8, formatted into `<agent_memory>` XML tags and inserted into the system prompt. Carries DC2 (memory content). No size limit on memory files (unlike skills).
- **Validation**: None. Content is decoded and interpolated directly via `str.format()` into `MEMORY_SYSTEM_PROMPT` at `middleware/memory.py:MemoryMiddleware.modify_request`. No content sanitization before injection.
- **Trust assumption**: Memory files are user-controlled; their content is trusted by the framework. The `MEMORY_SYSTEM_PROMPT` actively instructs the LLM to treat loaded content as authoritative and to update memory files immediately — amplifying the impact of any injected content.

#### DF3: SkillsMiddleware → LLM system prompt

- **Data**: SKILL.md content parsed with `yaml.safe_load` (frontmatter), then description/name/path injected into system prompt. Full body loaded on demand. Carries DC3.
- **Validation**: `yaml.safe_load` prevents code execution in YAML. Name length (64 chars), description length (1024 chars), 10MB file size check (`middleware/skills.py:_parse_skill_metadata`, `MAX_SKILL_FILE_SIZE`). Name validation via `_validate_skill_name` — but invalid names only produce a warning, they are **not rejected** (`middleware/skills.py` lines 302-308).
- **Trust assumption**: Skill source paths are user-configured; their content is trusted.

#### DF7: LLM → LocalShellBackend (`execute`)

- **Data**: LLM-generated shell command string, passed through `FilesystemMiddleware._create_execute_tool` to `subprocess.run(shell=True)`. Produces DC4 (shell output) that re-enters agent context via DF12.
- **Validation**: Command validated as non-empty string only (`backends/local_shell.py:LocalShellBackend.execute`). No command allow-listing, blocklisting, or pattern filtering. Timeout enforced (default 120s). Output truncated at 100KB.
- **Trust assumption**: Requires user to explicitly provide `LocalShellBackend` (not the default) and trust the LLM + HITL middleware.

#### DF11: Remote LangGraph Server → AsyncSubAgentMiddleware → agent context

- **Data**: Task run output (status, run result content, error messages), injected as `ToolMessage` into DC1. Content is not sanitized before injection.
- **Validation**: None. Run output from the remote server is inserted verbatim into the agent's conversation history. `subagent_type` is validated against registered names at launch time, but response content is not filtered.
- **Trust assumption**: Remote server is user-controlled and trusted.

---

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|------------|----------------|
| T1 | DF2, DF3 | DC2, DC3 | Context injection via poisoned memory or skill file — content injected verbatim into system prompt with no sanitization | TB4 | High | Verified | `middleware/memory.py:MemoryMiddleware.modify_request`, `middleware/skills.py:SkillsMiddleware.modify_request` |
| T2 | DF1 | DC1 | Prompt injection propagation through subagent `description` — LLM-controlled text becomes subagent's first HumanMessage | TB3 | Medium | Likely | `middleware/subagents.py:_build_task_tool` |
| T3 | DF7, DF12 | DC4 | Arbitrary shell command execution via LLM-controlled `execute` with `shell=True`; no command validation | TB5 | High | Verified | `backends/local_shell.py:LocalShellBackend.execute` |
| T4 | DF9 | DC1 | Unsafe deserialization of LangGraph checkpoint state (msgpack) — **fixed upstream** | TB4 | Medium | Disproven | `langgraph` dependency |
| T5 | DF3 | DC3 | Denial of service via oversized skill file flooding memory | TB4 | Low | Verified | `middleware/skills.py:MAX_SKILL_FILE_SIZE` |
| T6 | DF7 | DC4 | Host filesystem path escape via LocalShellBackend with `virtual_mode=True` — shell bypasses `_resolve_path` | TB5 | Medium | Verified | `backends/local_shell.py:LocalShellBackend.execute` |
| T7 | DF10 | DC1, DC6 | OpenAI Responses API retains conversation data by default when `openai:` model prefix is used | TB2 | Medium | Verified | `_models.py:resolve_model` |
| T8 | DF11 | DC1, DC7 | Remote LangGraph server response injected verbatim into main agent context (prompt injection via async subagent output) | TB6, TB3 | Medium | Likely | `middleware/async_subagents.py:AsyncSubAgentMiddleware` |
| T9 | DF2, DF3 | DC2, DC3 | FilesystemBackend `virtual_mode=None` default does not restrict file paths — `virtual_mode` exists primarily to support `CompositeBackend` path routing, not as a security boundary | TB5 | Info | Verified | `backends/filesystem.py:FilesystemBackend._resolve_path` |

### Threat Details

#### T1: Context Injection via Memory/Skill Files

- **Flow**: DF2 (MemoryMiddleware → LLM), DF3 (SkillsMiddleware → LLM)
- **Description**: If an attacker can write to a backend path used as a memory source (`memory=["..."]`) or skill source (`skills=["..."]`), they can inject arbitrary instructions into the agent's system prompt. For memory: content is decoded from UTF-8 and interpolated via `str.format()` into `MEMORY_SYSTEM_PROMPT` with zero validation — no size limit, no content filtering, no escaping. For skills: frontmatter description/name injected at load time (name validation only warns, does not reject invalid names); body injected when agent reads the file path. The `MEMORY_SYSTEM_PROMPT` amplifies risk by instructing the LLM to treat loaded content as authoritative and to immediately persist memory updates via `edit_file`.
- **Preconditions**: Attacker must control a file at a path the configured backend serves. With `FilesystemBackend`, this means filesystem write access to any configured `sources` path. Most practical vector: poisoned `.deepagents/AGENTS.md` committed to a git repository that a developer clones.

#### T2: Prompt Injection Propagation through Task Tool

- **Flow**: DF1 (LLM → SubAgentMiddleware)
- **Description**: If the main agent is compromised via prompt injection, an attacker-controlled `description` string is passed as the first `HumanMessage` to a subagent. The subagent inherits the same tool set and acts on this description autonomously. `subagent_type` is validated against registered names but `description` is not sanitized.
- **Preconditions**: Main agent must have already been compromised via prompt injection or adversarial context.

#### T3: Arbitrary Shell Command Execution (LocalShellBackend)

- **Flow**: DF7 (LLM → LocalShellBackend), DF12 (output → context)
- **Description**: `LocalShellBackend.execute` passes the LLM-generated command string directly to `subprocess.run(shell=True)` at `backends/local_shell.py:LocalShellBackend.execute`. Zero validation on command content beyond a non-empty string check. Commands execute with the process owner's full permissions. With `inherit_env=True`, all process environment variables (including API keys) are available to every command.
- **Preconditions**: User must explicitly configure `LocalShellBackend` as the backend (not the default — `StateBackend` is default, and it does not implement `SandboxBackendProtocol`, so the `execute` tool is filtered out at `middleware/filesystem.py:FilesystemMiddleware`). HITL middleware (`interrupt_on={"execute": True}`) is available but opt-in, not default.

#### T4: Unsafe msgpack Deserialization in LangGraph Checkpointer

- **Flow**: DF9 (checkpointer → agent state)
- **Description**: Unsafe msgpack deserialization previously identified in an upstream dependency. Confirmed fixed and closed upstream — see Investigated and Dismissed section.

#### T5: DoS via Oversized Skill File

- **Flow**: DF3 (SkillsMiddleware loading)
- **Description**: A SKILL.md file larger than 10MB is skipped rather than causing memory exhaustion. `MAX_SKILL_FILE_SIZE` checked at `middleware/skills.py:_parse_skill_metadata`. Note: memory files (AGENTS.md) have **no** size limit.
- **Preconditions**: Attacker must have write access to a skill source path.

#### T6: Path Restriction Bypass via Shell in LocalShellBackend

- **Flow**: DF7 (LLM → LocalShellBackend.execute)
- **Description**: Even when `virtual_mode=True` restricts file operation paths to `root_dir`, the `execute()` method runs shell commands without path restrictions. A command like `cat /etc/passwd` bypasses all `_resolve_path` guardrails. Documented explicitly in `LocalShellBackend` docstring.
- **Preconditions**: User must explicitly provide `LocalShellBackend`. Any shell access.

#### T7: OpenAI Responses API Conversation Data Retention

- **Flow**: DF10 (Framework → LLM via OpenAI Responses API)
- **Description**: `_models.py:resolve_model` detects the `openai:` prefix and initializes the model with `use_responses_api=True`. This causes implicit data retention on OpenAI servers for all conversation data including tool outputs.
- **Preconditions**: User passes `"openai:..."` model spec without explicit `store=False` override.

#### T8: Async Subagent Output Injected Verbatim into Main Agent Context

- **Flow**: DF11 (Remote LangGraph server → AsyncSubAgentMiddleware → DC1)
- **Description**: When an async subagent run completes, the output is fetched from the remote server and injected as a `ToolMessage` into conversation history without sanitization. If the remote subagent processes untrusted content, it can propagate prompt injection back to the main agent.
- **Preconditions**: User must configure at least one `AsyncSubAgent` spec. Remote server must be reachable.

#### T9: FilesystemBackend `virtual_mode=None` Default

- **Flow**: DF2, DF3 (backend reads)
- **Description**: `FilesystemBackend` defaults to `virtual_mode=None` → `False`, which does not restrict file operation paths. The `virtual_mode` flag exists primarily to support `CompositeBackend` path routing (mapping path prefixes to different backends), not as a security boundary. A deprecation warning is issued; the default flips to `True` in v0.5.0.
- **Preconditions**: User opts into `FilesystemBackend` without specifying `virtual_mode=True`.

---

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (API params) | DF4, DF5, DF6 | — | `StoreBackend._validate_namespace` (namespace chars only) | User | No validation of backend paths, model strings, skill/memory source paths |
| LLM output (tool calls) | DF1, DF7, DF10 | T2, T3, T8 | `subagent_type` allowlist; command non-empty check | Framework (allowlist); User (content) | No sanitization of `description`, shell command content, or async task payloads |
| Tool/function results (re-entering context) | DF1, DF8, DF11, DF12 | T1, T2, T8 | `_EXCLUDED_STATE_KEYS` (inline subagents only) | Framework (key filtering); User (content) | Tool results and async subagent output injected verbatim into conversation history |
| Backend-stored files (memory, skills) | DF2, DF3 | T1, T5 | `yaml.safe_load`; size cap 10MB (skills only); description 1024 chars | User (file content trust) | No content sanitization before system prompt injection; **no size limit on memory files** |
| Configuration (checkpointer, store) | DF9 | T4 (fixed) | None (delegated to user implementations) | User | Transitive langgraph msgpack — now fixed upstream |
| Remote LangGraph server responses | DF11 | T8 | `subagent_type` allowlist at launch; TLS (SDK-managed) | User (server trustworthiness) | Run output not sanitized; no server identity verification beyond TLS |
| Model provider (OpenAI Responses API) | DF10 (outbound) | T7 | None (opt-out via user model config) | User (opt-out) | `use_responses_api=True` default on `openai:` prefix causes implicit retention |

---

## Out-of-Scope Threats

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to harmful actions via user-registered tools | The project does not control which tools the user registers or what those tools do. Default backend (`StateBackend`) does not support shell execution. | Providing safe defaults; documenting risks in `LocalShellBackend` and `FilesystemBackend` docstrings |
| LLM model behavior (refusals, jailbreaks, output safety) | The project cannot guarantee safety across all models. `resolve_model` accepts any `BaseChatModel`. | `_models.py:resolve_model` — after model resolution, model behavior is external |
| Deployment security (HTTPS, auth, network access controls) | The library does not run a server. Users control how they expose the compiled agent graph. | Library produces a `CompiledStateGraph`; deployment is user-controlled |
| S3/database/checkpointer backend security | Users configure their own checkpointer and store. The library passes them through unchanged. | `graph.py:create_deep_agent` — `checkpointer` and `store` are passed through |
| Supply chain attacks on LangChain/LangGraph/provider libraries | Transitive dependencies are managed by the user's resolver. | `pyproject.toml` — only lower-bound version pins |
| Credential leakage from user-controlled system prompts or AGENTS.md | Memory guidelines instruct agent not to store API keys (`middleware/memory.py:MEMORY_SYSTEM_PROMPT`). User-stored credentials in memory files is user misconfiguration. | `MEMORY_SYSTEM_PROMPT` instruction; file content trust is the user's responsibility |
| Remote LangGraph server compromise affecting async subagent output | Remote server is user-provisioned infrastructure. Library uses TLS and validates `subagent_type` before connecting. | `middleware/async_subagents.py:AsyncSubAgentMiddleware` — after client construction, server behavior is external |
| OpenAI data retention when user explicitly configures `store=False` | If user provides pre-initialized model with `store=False`, library passes through unchanged. | `_models.py:resolve_model` — auto-selection is the boundary |

### Rationale

**Prompt injection + user tool = code execution**: The framework ships `StateBackend` as default — no shell execution possible without explicit opt-in to `LocalShellBackend`. The `execute` tool is filtered out by `FilesystemMiddleware` when the backend does not implement `SandboxBackendProtocol` (`middleware/filesystem.py`). When users opt into `LocalShellBackend`, HITL middleware is available but requires explicit `interrupt_on` configuration.

**Memory/skill content injection**: Memory and skill files are user-controlled artifacts. The framework treats them as trusted — this is documented. The primary risk vector is a poisoned `.deepagents/AGENTS.md` in a cloned git repository. This is structurally identical to the risk of running `pyproject.toml` build scripts or `Makefile` targets from an untrusted repo.

**AsyncSubAgent server trust**: `AsyncSubAgent` entries are user-declared with specific `url` and `graph_id` values. The user explicitly chooses to trust a remote server.

---

## Investigated and Dismissed

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | Unsafe msgpack deserialization in LangGraph checkpointer | Verified fix status — confirmed fixed and closed upstream. | Users on current `langgraph` versions are not exposed. | Upstream langgraph has patched the unsafe deserialization. Not an active risk. Moved T4 status to Disproven. |
| D2 | YAML code execution in skill frontmatter parsing | Traced `middleware/skills.py:_parse_skill_metadata` — uses `yaml.safe_load()` exclusively. | `middleware/skills.py:_parse_skill_metadata` | `yaml.safe_load()` prevents arbitrary Python code execution during YAML parsing. Not exploitable. Note: parsed string values can still carry prompt injection payloads (covered by T1). |
| D3 | BaseSandbox shell injection via file operation commands | Traced `backends/sandbox.py` — `_GLOB_COMMAND_TEMPLATE`, `_WRITE_CHECK_TEMPLATE`, and edit operations use base64-encoded parameters decoded inside Python scripts, not string interpolation into shell. | `backends/sandbox.py:_GLOB_COMMAND_TEMPLATE` — `base64.b64decode('{path_b64}').decode('utf-8')` pattern | Base64 encoding avoids shell metacharacter injection. User-controlled paths are encoded before interpolation into command templates. Not exploitable via shell injection. |

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-10 | langster-threat-model (automated) | Initial threat model for SDK library (libs/deepagents); scoped to library only, CLI excluded |
| 2026-03-27 | langster-threat-model (automated, deep mode) | Deep expansion: added C12, TB6, DF10-DF11, DC1-DC7, T7-T9; updated T4; added Investigated and Dismissed |
| 2026-03-28 | langster-threat-model (automated, deep mode) | Deep validation pass: **removed Status column and Mitigations/Residual Risk fields** (open source visibility compliance); added Validation column with flaw validation for all threats; T1 validated as Verified (zero content sanitization on memory path, skill name rejection is warning-only); T3 validated as Verified (command passes to shell=True with no filtering, but not default backend); added C13 (BaseSandbox), C14 (FilesystemMiddleware); added DF12 (shell output re-entering context); added D2 (YAML safe_load dismissed), D3 (BaseSandbox base64 encoding dismissed); fixed Data Flow table format (Classification and Protocol columns); added Data Classification Details for DC1, DC4, DC5; updated T4 validation to Disproven |
