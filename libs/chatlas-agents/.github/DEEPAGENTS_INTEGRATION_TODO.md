# DeepAgents CLI Integration TODO

This document outlines the plan to extend ChATLAS CLI by wrapping and integrating key features from deepagents-cli.

## Overview

**Goal**: Enhance ChATLAS CLI with:
- Persistent memory system (user + project context)
- Progressive skill disclosure for ATLAS-specific workflows
- Human-in-the-loop approval for sensitive operations
- Rich interactive interface with planning capabilities

**Strategy**: Wrapper approach - extend ChATLAS CLI while preserving existing functionality

---

## Phase 1: Memory System Implementation

### 1.1 Create Memory Module Structure

- [ ] Create `chatlas_agents/memory/` directory
- [ ] Create `chatlas_agents/memory/__init__.py`
- [ ] Create `chatlas_agents/memory/user_memory.py`
- [ ] Create `chatlas_agents/memory/project_memory.py`
- [ ] Create `chatlas_agents/memory/middleware.py`
- [ ] Create `chatlas_agents/memory/loader.py`

### 1.2 Implement User Memory

**File**: `chatlas_agents/memory/user_memory.py`

- [ ] Implement `get_user_memory_dir()` function
  - Returns `~/.chatlas/agents/<agent_name>/`
  - Creates directory if it doesn't exist
- [ ] Implement `get_user_agent_md_path(agent_name: str)` function
  - Returns path to `~/.chatlas/agents/<agent_name>/agent.md`
- [ ] Implement `load_user_memory(agent_name: str)` function
  - Reads and returns content of user agent.md
  - Returns empty string if file doesn't exist
- [ ] Implement `save_user_memory(agent_name: str, content: str)` function
  - Writes content to user agent.md
- [ ] Implement `create_default_user_memory(agent_name: str)` function
  - Creates default agent.md with ATLAS-specific template
  - Template includes: personality, coding style, ATLAS conventions

### 1.3 Implement Project Memory

**File**: `chatlas_agents/memory/project_memory.py`

- [ ] Implement `find_project_root(start_path: Path | None)` function
  - Walks up directory tree looking for `.git`
  - Returns project root Path or None
- [ ] Implement `get_project_memory_paths(project_root: Path)` function
  - Returns list of paths: `.chatlas/agent.md` and `agent.md`
  - Only returns paths that exist
- [ ] Implement `load_project_memory(project_root: Path)` function
  - Loads all project agent.md files
  - Combines content from both locations if both exist
- [ ] Implement `list_project_memory_files(project_root: Path)` function
  - Lists all .md files in `.chatlas/` directory
  - Used for progressive memory loading

### 1.4 Implement Memory Middleware

**File**: `chatlas_agents/memory/middleware.py`

- [ ] Create `ChatLASMemoryMiddleware` class
  - Based on deepagents-cli's `AgentMemoryMiddleware`
  - Adapts to LangGraph middleware interface
- [ ] Implement memory injection into system prompt
  - Adds `<user_memory>` section from user agent.md
  - Adds `<project_memory>` section from project agent.md
  - Adds instructions on when to check/update memories
- [ ] Implement memory update instructions
  - When to update memories (feedback, new patterns)
  - How to decide user vs project memory
  - Learning from corrections pattern
- [ ] Add support for on-demand memory file loading
  - Agent can read `.chatlas/architecture.md`, etc.
  - Instructions on checking memory files first

### 1.5 Memory Templates

**File**: `chatlas_agents/memory/templates.py`

- [ ] Create `DEFAULT_USER_MEMORY_TEMPLATE` constant
  - ATLAS agent personality
  - Preferred coding style (Python type hints, etc.)
  - ATLAS software conventions
  - Tool preferences (ChATLAS MCP for documentation)
- [ ] Create `DEFAULT_PROJECT_MEMORY_TEMPLATE` constant
  - Project structure guidance
  - Team conventions
  - Common workflows
- [ ] Create `MEMORY_INSTRUCTIONS_TEMPLATE` constant
  - Instructions injected into system prompt
  - When to check memories
  - When to update memories
  - User vs project memory decision guide

---

## Phase 2: Skills System Implementation

### 2.1 Create Skills Module Structure

- [ ] Create `chatlas_agents/skills/` directory
- [ ] Create `chatlas_agents/skills/__init__.py`
- [ ] Create `chatlas_agents/skills/loader.py`
- [ ] Create `chatlas_agents/skills/middleware.py`
- [ ] Create `chatlas_agents/skills/builtin/` directory

### 2.2 Implement Skills Loader

**File**: `chatlas_agents/skills/loader.py`

- [ ] Adapt `_parse_skill_metadata()` from deepagents-cli
  - Parse YAML frontmatter (name, description)
  - Extract from SKILL.md files
  - Security checks (file size, safe paths)
- [ ] Implement `list_skills(skills_dir: Path, source: str)` function
  - Scans directory for skill subdirectories
  - Returns list of SkillMetadata
  - Source: "user" or "project"
- [ ] Implement `load_all_skills(agent_name: str)` function
  - Loads from `~/.chatlas/agents/<agent_name>/skills/`
  - Loads from `.chatlas/skills/` if in project
  - Combines both sources
- [ ] Implement `get_skill_content(skill_path: Path)` function
  - Reads full SKILL.md content
  - Used when agent needs detailed instructions

### 2.3 Implement Skills Middleware

**File**: `chatlas_agents/skills/middleware.py`

- [ ] Create `ChatLASSkillsMiddleware` class
  - Progressive disclosure pattern
  - Injects skill list into system prompt
  - Provides tool for reading skill content
- [ ] Implement skill list injection
  - Format: "Available Skills: ami-query - Query ATLAS AMI database, ..."
  - Only includes names and descriptions (not full content)
- [ ] Implement skill content access
  - Agent uses `read_file` to access full SKILL.md
  - Or provide dedicated `get_skill_content` tool
- [ ] Add instructions for skill usage
  - When to check available skills
  - How to load full skill instructions
  - How to follow skill workflows

### 2.4 Create ATLAS-Specific Skills

**Directory**: `chatlas_agents/skills/builtin/`

> **Note**: AMI Query and Rucio Access skills are **LOW PRIORITY** and marked as future work. These require implementing new MCP tools with the ATLAS software stack. A local MCP server will be needed to handle these integrations. Focus on the other skills first.

#### Skill 1: Paper Review (Priority)
- [ ] Create `chatlas_agents/skills/builtin/paper-review/` directory
- [ ] Create `paper-review/SKILL.md`
  - YAML frontmatter: name, description
  - When to use (reviewing ATLAS papers)
  - ATLAS style guide checklist
  - Structure verification steps
  - Common issues to check
- [ ] Create `paper-review/checklist.md` (supporting file)
  - Detailed ATLAS paper checklist
  - Figure requirements
  - Reference formatting
  - Collaboration approval process

#### Skill 2: ChATLAS Documentation (Priority)
- [ ] Create `chatlas_agents/skills/builtin/chatlas-docs/` directory
- [ ] Create `chatlas-docs/SKILL.md`
  - YAML frontmatter: name, description
  - When to use ChATLAS RAG search
  - Which vectorstores to query
  - How to interpret and cite results
  - Date filtering for outdated info

#### Skill 3: Analysis Workflow (Priority)
- [ ] Create `chatlas_agents/skills/builtin/analysis-workflow/` directory
- [ ] Create `analysis-workflow/SKILL.md`
  - YAML frontmatter: name, description
  - Standard ATLAS analysis steps
  - Dataset selection → Processing → Plotting → Review
  - HTCondor job submission for heavy tasks
  - Result verification

#### Future Skills (Low Priority - Requires Local MCP Server)

> **Note**: The following skills require implementing new MCP tools that interface with the ATLAS software stack. This is planned as a separate project phase involving a local MCP server implementation.

- [ ] **AMI Query Skill** - Query ATLAS AMI database for dataset information
  - Requires: AMI client integration, CERN authentication
  - MCP tools: `query_ami`, `list_datasets`, `get_provenance`
  - Location: `chatlas_agents/skills/builtin/ami-query/` (future)
- [ ] **Rucio Access Skill** - Access and manage ATLAS datasets via Rucio
  - Requires: Rucio client integration, grid certificates
  - MCP tools: `find_dataset`, `replicate_dataset`, `get_dataset_info`
  - Location: `chatlas_agents/skills/builtin/rucio-access/` (future)
- [ ] **Indico Meetings Skill** - Query ATLAS Indico for upcoming relevant meetings
  - Requires: Indico API integration
  - MCP tools: `search_indico`, `get_upcoming_meetings`
  - Location: `chatlas_agents/skills/builtin/indico-meetings/` (future)

### 2.5 Skill Installation System

**File**: `chatlas_agents/skills/installer.py`

- [ ] Implement `install_builtin_skills(agent_name: str)` function
  - Copies builtin skills to user directory
  - Creates `~/.chatlas/agents/<agent_name>/skills/`
  - Only installs if not already present
- [ ] Implement `list_builtin_skills()` function
  - Returns available builtin skills
- [ ] Implement `install_skill(agent_name: str, skill_name: str)` function
  - Installs specific builtin skill

---

## Phase 3: Enhanced CLI Commands

### 3.1 Interactive Command

**File**: `chatlas_agents/cli.py`

- [ ] Add `@app.command()` for `interactive` command
  - Arguments: `--agent`, `--auto-approve`, `--project`, `--config`
- [ ] Implement `interactive()` function
  - Loads configuration (env or YAML)
  - Creates agent with memory + skills middleware
  - Launches interactive REPL loop
- [ ] Create `_run_interactive_deep()` async function
  - Similar to `_run_interactive()` but enhanced
  - Show splash screen with memory/skills info
  - Display available slash commands
  - Handle `/memory`, `/skills`, `/tokens`, `/clear`, `/help`
  - Rich formatting for responses
- [ ] Add conversation state management
  - Track thread_id for persistence
  - Save/load conversation checkpoints
- [ ] Implement streaming support
  - Stream agent responses token-by-token
  - Show thinking/planning steps

### 3.2 Memory Management Commands

**File**: `chatlas_agents/cli.py`

- [ ] Add `@app.group()` for `memory` command group
- [ ] Implement `memory list` subcommand
  - Lists user and project memory files
  - Shows paths and last modified dates
  - Preview first few lines
- [ ] Implement `memory edit` subcommand
  - Opens user or project agent.md in editor
  - Options: `--type=user|project`, `--agent=<name>`
- [ ] Implement `memory show` subcommand
  - Displays full content of memory file
  - Formatted with syntax highlighting
- [ ] Implement `memory reset` subcommand
  - Resets to default template
  - Confirmation prompt
  - Backup old version
- [ ] Implement `memory create` subcommand
  - Creates new project memory file
  - Options: `--name=<filename>`, `--template=<type>`
  - For creating architecture.md, deployment.md, etc.

### 3.3 Skills Management Commands

**File**: `chatlas_agents/cli.py`

- [ ] Add `@app.group()` for `skills` command group
- [ ] Implement `skills list` subcommand
  - Lists all available skills (user + project + builtin)
  - Shows: name, description, source
  - Options: `--project` (only project), `--user` (only user), `--builtin`
- [ ] Implement `skills info` subcommand
  - Shows detailed information about a skill
  - Displays full SKILL.md content
  - Arguments: skill name
  - Options: `--project`, `--user`
- [ ] Implement `skills create` subcommand
  - Creates new skill from template
  - Arguments: skill name
  - Options: `--project` (create in project), `--from-template=<name>`
  - Interactive prompts for name, description
- [ ] Implement `skills install` subcommand
  - Installs builtin skill to user directory
  - Arguments: skill name or "all"
  - Options: `--agent=<name>`
- [ ] Implement `skills search` subcommand
  - Searches skills by keyword
  - Searches in names, descriptions, and content

### 3.4 Enhanced Splash Screen

**File**: `chatlas_agents/cli.py`

- [ ] Update `show_splash()` function
  - Add memory status (user memory loaded)
  - Add project detection (in project: <name>)
  - Add skills count (N skills available)
  - Add available tools summary
- [ ] Create `show_session_info()` function
  - Display at start of interactive session
  - Show active agent name
  - Show thread ID
  - Show memory sources
  - Show loaded skills

---

## Phase 4: Human-in-the-Loop Integration

### 4.1 Create HITL Module

**File**: `chatlas_agents/hitl/__init__.py`

- [ ] Create `chatlas_agents/hitl/` directory
- [ ] Import and adapt approval functions from deepagents-cli
  - `prompt_for_tool_approval()` with arrow key navigation
  - `ApproveDecision`, `RejectDecision` classes

### 4.2 Implement File Operation Approval

**File**: `chatlas_agents/hitl/file_ops.py`

- [ ] Create `FileOpPreview` class
  - Title, details, diff, error fields
- [ ] Implement `build_file_approval_preview()` function
  - For write_file: show create/overwrite, line count
  - For edit_file: show diff of changes
  - For shell commands: show command and working dir
- [ ] Implement `format_diff()` function
  - Generate unified diff for file changes
  - Syntax highlighting
- [ ] Create approval middleware
  - Intercepts file write/edit operations
  - Shows preview and prompts for approval
  - Handles rejection gracefully

### 4.3 Implement Tool Approval System

**File**: `chatlas_agents/hitl/middleware.py`

- [ ] Create `ChatLASHITLMiddleware` class
  - Wraps sensitive operations
  - Configurable approval requirements
- [ ] Define sensitive operations
  - File writes/edits (always require approval unless --auto-approve)
  - Shell commands (always require approval)
  - MCP tool calls (configurable per tool)
  - HTCondor submissions (always require approval)
- [ ] Implement approval UI
  - Use Rich panels for preview
  - Arrow key navigation (approve/reject/auto-accept-all)
  - Handle Ctrl+C gracefully
- [ ] Implement rejection handling
  - Log rejection reason
  - Agent acknowledges and suggests alternatives
  - Never retry exact same command

### 4.4 Add Approval to HTCondor Submission

**File**: `chatlas_agents/htcondor.py`

- [ ] Add HITL approval before job submission
  - Show job details: name, resources, prompt
  - Show estimated cost/time
  - Preview submit file content
- [ ] Add `--dry-run` support (skip approval)
- [ ] Add confirmation after submission
  - Show job ID
  - Show monitoring commands

---

## Phase 5: UI/UX Enhancements

### 5.1 Token Tracking

**File**: `chatlas_agents/ui/tokens.py`

- [ ] Create `TokenTracker` class (from deepagents-cli)
  - Track input/output tokens per message
  - Track total session usage
  - Calculate costs per model
- [ ] Implement `display_session()` method
  - Show total tokens used
  - Show estimated cost
  - Show tokens per message
- [ ] Add to interactive mode
  - Display after each response (optional)
  - `/tokens` command shows full breakdown

### 5.2 Rich Console Formatting

**File**: `chatlas_agents/ui/formatting.py`

- [ ] Implement `render_diff_block()` function
  - Syntax-highlighted diffs
  - Side-by-side or unified view
- [ ] Implement `render_file_operation()` function
  - Format file operations nicely
  - Show before/after preview
- [ ] Implement `render_todo_list()` function
  - Checkbox display for todos
  - Status colors (not-started, in-progress, completed)
  - Progress bar
- [ ] Implement `render_tool_call()` function
  - Format tool invocations
  - Show arguments nicely
  - Syntax highlighting for code

### 5.3 Image Support

**File**: `chatlas_agents/ui/images.py`

- [ ] Implement `ImageTracker` class (from deepagents-cli)
  - Track image attachments in conversation
  - Store base64 encoded images
- [ ] Implement `parse_file_mentions()` function
  - Parse `[image]`, `[image 1]` placeholders
  - Load images from paths
- [ ] Implement `create_multimodal_content()` function
  - Create LangChain message with images
  - Base64 encode images
- [ ] Add to interactive input
  - Detect image paths in input
  - Auto-attach images to messages

---

## Phase 6: Configuration and Setup

### 6.1 Update Configuration

**File**: `chatlas_agents/config/__init__.py`

- [ ] Add `MemoryConfig` class
  - `agent_name: str`
  - `enable_user_memory: bool`
  - `enable_project_memory: bool`
  - `auto_create_default: bool`
- [ ] Add `SkillsConfig` class
  - `enable_skills: bool`
  - `builtin_skills: List[str]`
  - `auto_load_project_skills: bool`
- [ ] Add `HITLConfig` class
  - `enabled: bool`
  - `auto_approve: bool`
  - `require_approval_for: List[str]`
- [ ] Update `AgentConfig` to include new configs
  - `memory: MemoryConfig`
  - `skills: SkillsConfig`
  - `hitl: HITLConfig`

### 6.2 Update Environment Variables

**File**: `.env.example`

- [ ] Add memory-related variables
  - `CHATLAS_AGENT_NAME=atlas-agent`
  - `CHATLAS_ENABLE_MEMORY=true`
  - `CHATLAS_ENABLE_SKILLS=true`
- [ ] Add HITL variables
  - `CHATLAS_AUTO_APPROVE=false`
  - `CHATLAS_REQUIRE_APPROVAL=write_file,edit_file,shell`

### 6.3 Update Agent Creation

**File**: `chatlas_agents/agents/__init__.py`

- [ ] Update `create_deep_agent()` function
  - Add memory middleware if enabled
  - Add skills middleware if enabled
  - Add HITL middleware if enabled
  - Load and inject MCP tools
- [ ] Implement `create_interactive_agent()` function
  - Specialized for interactive mode
  - All middleware enabled by default
  - Enhanced system prompt
- [ ] Add middleware composition logic
  - Proper ordering: Memory → Skills → HITL → Agent
  - Handle conflicts/dependencies

---

## Phase 7: Documentation and Examples

### 7.1 Update Main README

**File**: `README.md`

- [ ] Add "Interactive Mode" section
  - Describe new `chatlas-agents interactive` command
  - Explain memory system
  - Explain skills system
  - Show example session
- [ ] Add "Memory Management" section
  - How to view/edit memories
  - User vs project memory
  - Memory update patterns
- [ ] Add "Skills System" section
  - Available builtin skills
  - How to create custom skills
  - SKILL.md format
- [ ] Add "Human-in-the-Loop" section
  - What requires approval
  - How to use --auto-approve
  - Security best practices

### 7.2 Create Interactive Mode Guide

**File**: `docs/INTERACTIVE_MODE.md`

- [ ] Getting started with interactive mode
- [ ] Available commands and shortcuts
- [ ] Working with memory
- [ ] Using skills
- [ ] Approval workflow
- [ ] Tips and best practices

### 7.3 Create Skills Development Guide

**File**: `docs/SKILLS_DEVELOPMENT.md`

- [ ] SKILL.md format specification
- [ ] YAML frontmatter requirements
- [ ] Writing effective skill instructions
- [ ] Including supporting files
- [ ] Testing skills
- [ ] Sharing skills with team

### 7.4 Create Memory Management Guide

**File**: `docs/MEMORY_MANAGEMENT.md`

- [ ] Memory system architecture
- [ ] User memory vs project memory
- [ ] When agent updates memories
- [ ] Best practices for memory organization
- [ ] Progressive memory loading pattern
- [ ] Example memory files

### 7.5 Create Examples

**Directory**: `examples/interactive/`

- [ ] Create `basic_interactive.py`
  - Simple interactive session example
  - Memory and skills disabled
- [ ] Create `with_memory.py`
  - Interactive with memory enabled
  - Shows memory loading/updating
- [ ] Create `with_skills.py`
  - Interactive with ATLAS skills
  - Demonstrates skill usage
- [ ] Create `full_featured.py`
  - All features enabled
  - Complete workflow example

---

## Phase 8: Testing

### 8.1 Unit Tests

**Directory**: `tests/`

- [ ] Create `tests/test_memory.py`
  - Test memory loading/saving
  - Test project detection
  - Test memory middleware
- [ ] Create `tests/test_skills.py`
  - Test skill loading
  - Test skill metadata parsing
  - Test progressive disclosure
- [ ] Create `tests/test_hitl.py`
  - Test approval workflow
  - Test rejection handling
  - Test auto-approve mode
- [ ] Create `tests/test_cli_interactive.py`
  - Test interactive command
  - Test slash commands
  - Test memory/skills commands

### 8.2 Integration Tests

**Directory**: `tests/integration/`

- [ ] Create `tests/integration/test_interactive_session.py`
  - Full interactive session flow
  - Memory persistence across sessions
  - Skills usage
- [ ] Create `tests/integration/test_project_memory.py`
  - Project detection
  - Project memory loading
  - Project skills loading
- [ ] Create `tests/integration/test_atlas_skills.py`
  - Test each core ATLAS skill (Paper Review, ChATLAS Docs, Analysis Workflow)
  - Verify skill instructions are correct
  - Test skill workflows
  - Skip AMI/Rucio skills (future work)

---

## Phase 9: Migration and Backwards Compatibility

### 9.1 Preserve Existing Functionality

- [ ] Ensure existing `run` command still works
  - No breaking changes to current interface
  - New features are opt-in
- [ ] Ensure `htcondor-submit` command still works
  - Add HITL approval as enhancement
  - Keep existing flags working

### 9.2 Create Migration Guide

**File**: `docs/MIGRATION_GUIDE.md`

- [ ] Document changes from v0.1.0 to v0.2.0
- [ ] How to migrate to interactive mode
- [ ] How to set up memory for existing agents
- [ ] How to adopt skills system

---

## Phase 10: Deployment and Release

### 10.1 Version Bump

- [ ] Update version in `pyproject.toml` to `0.2.0`
- [ ] Update version in `chatlas_agents/__init__.py`
- [ ] Update changelog with new features

### 10.2 Dependencies

- [ ] Add new dependencies to `requirements.txt` and `pyproject.toml`
  - `prompt-toolkit` (for interactive input)
  - `markdownify` (for URL fetching)
  - `tavily-python` (optional, for web search)
- [ ] Update Docker image to include new dependencies

### 10.3 Release Preparation

- [ ] Create comprehensive CHANGELOG.md entry
- [ ] Update screenshots/demos in README
- [ ] Create demo video of interactive mode
- [ ] Tag release: v0.2.0

---

## Success Criteria

### Functionality
- [ ] Interactive mode launches successfully
- [ ] Memory system loads and saves correctly
- [ ] Skills are loaded and usable
- [ ] HITL approval works for sensitive operations
- [ ] All existing features still work

### User Experience
- [ ] Clean, intuitive CLI interface
- [ ] Fast startup time (<5 seconds)
- [ ] Responsive interactive session
- [ ] Helpful error messages
- [ ] Good documentation coverage

### Code Quality
- [ ] 80%+ test coverage
- [ ] Type hints for all public APIs
- [ ] No regressions in existing functionality
- [ ] Passes all CI/CD checks

---

## Future Work - Local MCP Server for ATLAS Tools

The following items are outside the current project scope and will be addressed in a future phase:

### Local MCP Server Implementation
- [ ] Design local MCP server architecture
- [ ] Implement MCP server with ATLAS software stack integration
- [ ] Create AMI query tools (`query_ami`, `list_datasets`, `get_provenance`)
- [ ] Create Rucio access tools (`find_dataset`, `replicate_dataset`, `get_dataset_info`)
- [ ] Create Indico integration tools (`search_indico`, `get_upcoming_meetings`)
- [ ] Set up authentication and credential management
- [ ] Deploy local MCP server on lxplus or similar CERN infrastructure
- [ ] Update ChATLAS agent configuration to connect to local MCP server
- [ ] Implement AMI Query and Rucio Access skills once tools are available

**Rationale**: These features require deep integration with the ATLAS software stack, which is better handled through a dedicated MCP server rather than direct client integration. This allows for:
- Centralized authentication and credential management
- Separation of concerns (agent logic vs ATLAS tool integration)
- Easier testing and maintenance
- Potential reuse by other ATLAS AI tools

---

## Notes

- **Phased Rollout**: Can implement phases incrementally, each providing value
- **Testing**: Test each phase thoroughly before moving to next
- **Documentation**: Update docs alongside code changes
- **Backwards Compatibility**: Preserve all existing CLI functionality
- **ATLAS Focus**: Customize for ATLAS workflows, not generic use case
- **Scope Management**: AMI/Rucio integration deferred to future local MCP server project

---

## Quick Start for Contributors

1. **Start with Phase 1**: Memory system is foundation for other features
2. **Test thoroughly**: Use `uv run pytest` after each change
3. **Follow patterns**: Adapt from deepagents-cli but customize for ChATLAS
4. **Update docs**: Keep documentation in sync with code

---

## Appendix: DeepAgents CLI Architecture and Dependencies

### Overview of DeepAgents CLI

The **deepagents-cli** is an interactive command-line interface built on top of the DeepAgents framework. It provides a terminal-based coding assistant similar to Claude Code, with persistent memory, skills system, and human-in-the-loop controls.

**Repository**: [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)  
**Package**: `deepagents-cli` (part of the DeepAgents monorepo)

### Core Functionality

#### 1. Interactive REPL Loop
- Rich console interface with syntax highlighting
- Streaming responses from LLM
- Slash commands (`/help`, `/clear`, `/tokens`, `/quit`)
- Bash command execution (`!command`)
- Image support in prompts

**Implementation**: `deepagents_cli/main.py::simple_cli()`

#### 2. Memory System
- **User Memory**: `~/.deepagents/<agent_name>/agent.md` - Personal preferences and coding style
- **Project Memory**: `.deepagents/agent.md` in project root - Project-specific conventions
- **Progressive Loading**: Additional memory files loaded on-demand
- **Auto-Updates**: Agent updates memories based on feedback and patterns

**Implementation**: `deepagents_cli/agent_memory.py::AgentMemoryMiddleware`

**Key Features**:
- Detects `.git` directory for project root identification
- Hierarchical loading (user → project)
- Injected into system prompt as `<user_memory>` and `<project_memory>` tags
- Instructions on when/how to update memories

#### 3. Skills System
- **Progressive Disclosure**: Skills listed by name/description in system prompt
- **On-Demand Loading**: Full skill content loaded only when needed
- **SKILL.md Format**: YAML frontmatter + Markdown instructions
- **Supporting Files**: Skills can include scripts, configs, etc.

**Implementation**: `deepagents_cli/skills/`

**Skill Locations**:
- Global: `~/.deepagents/<agent_name>/skills/`
- Project: `.deepagents/skills/` in project root
- Builtin: Provided as examples in repository

**Example Skills**:
- `web-research`: Structured research workflow with parallel delegation
- `langgraph-docs`: LangGraph documentation lookup

#### 4. Human-in-the-Loop (HITL)
- **Approval Required For**: File writes/edits, shell commands, web searches, subagent spawning
- **Interactive UI**: Arrow-key navigation for approve/reject/auto-accept-all
- **Rich Previews**: Diffs for file changes, command details
- **Rejection Handling**: Agent acknowledges and suggests alternatives

**Implementation**: 
- `deepagents_cli/execution.py::prompt_for_tool_approval()`
- `deepagents_cli/file_ops.py::build_approval_preview()`

#### 5. Built-in Tools
- **File Operations**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Command Execution**: `shell` (local), `execute` (remote sandbox)
- **Web Integration**: `web_search` (Tavily API), `fetch_url`, `http_request`
- **Planning**: `write_todos` for task management
- **Delegation**: `task` for spawning subagents

**Implementation**: `deepagents_cli/tools.py`

#### 6. Sandbox Support
- **Local Mode**: Runs commands directly on host
- **Remote Sandboxes**: Modal, Runloop, Daytona integrations
- **Isolation**: Secure execution environment for untrusted code

**Implementation**: `deepagents_cli/integrations/sandbox_factory.py`

### Architecture Stack

```
┌─────────────────────────────────────────┐
│         deepagents-cli (CLI Layer)      │
│  • REPL loop and user interface         │
│  • Memory system (AgentMemoryMiddleware)│
│  • Skills system (SkillsMiddleware)     │
│  • HITL approvals                       │
│  • Shell execution (ShellMiddleware)    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       deepagents (Agent Framework)      │
│  • create_deep_agent() factory          │
│  • TodoListMiddleware (planning)        │
│  • FilesystemMiddleware (mock FS)      │
│  • SubAgentMiddleware (delegation)      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         LangGraph (Orchestration)       │
│  • State graph management               │
│  • Middleware system                    │
│  • Checkpointing (conversation memory)  │
│  • Runtime execution                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         LangChain (Foundation)          │
│  • LLM abstractions (ChatModel)         │
│  • Tool definitions (BaseTool)          │
│  • Message types                        │
│  • Callbacks and tracing                │
└─────────────────────────────────────────┘
```

### Key Dependencies

#### DeepAgents Core (`deepagents` package)
**Repository**: [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/tree/master/libs/deepagents)

**Key Exports**:
- `create_deep_agent()` - Main factory function for creating deep agents
- `TodoListMiddleware` - Planning and task management
- `FilesystemMiddleware` - Mock file system for agent operations
- `SubAgentMiddleware` - Spawns isolated subagents for subtasks

**Purpose**: Provides the core agent capabilities (planning, file operations, subagent spawning) that deepagents-cli builds upon.

**LangChain Docs**: [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview)

#### LangGraph (`langgraph` package)
**Repository**: [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)

**Key Features Used**:
- **State Graphs**: Orchestrates agent execution flow
- **Middleware System**: Composable agent capabilities
- **Checkpointing**: Persists conversation state across sessions
- **Human-in-the-Loop**: Interrupt mechanism for approvals
- **Runtime**: Executes agent graphs with configuration

**Purpose**: Provides the orchestration layer that manages agent state, execution flow, and middleware composition.

**LangChain Docs**: [LangGraph Introduction](https://docs.langchain.com/langgraph)

#### LangChain Core (`langchain-core` package)
**Repository**: [langchain-ai/langchain](https://github.com/langchain-ai/langchain)

**Key Features Used**:
- **Language Models**: `BaseChatModel` abstraction for LLMs
- **Tools**: `BaseTool` interface for agent capabilities
- **Messages**: Structured message types (HumanMessage, AIMessage, ToolMessage)
- **Runnables**: Composable execution units
- **Callbacks**: Tracing and observability

**Purpose**: Foundation layer providing LLM, tool, and message abstractions.

**LangChain Docs**: [LangChain Core Concepts](https://docs.langchain.com/concepts)

#### LangChain Providers
- **`langchain-openai`**: OpenAI model integration (GPT-4, GPT-3.5)
- **`langchain-anthropic`**: Anthropic model integration (Claude 3.5 Sonnet)
- **`langchain-google-genai`**: Google model integration (Gemini)

**Purpose**: LLM provider implementations for different model backends.

### Middleware System

The deepagents-cli extends the base DeepAgents middleware with CLI-specific capabilities:

```python
# DeepAgents Core Middleware (built-in)
TodoListMiddleware        # Planning and task management
FilesystemMiddleware      # Mock file system operations  
SubAgentMiddleware        # Subagent spawning and delegation

# DeepAgents CLI Middleware (custom)
AgentMemoryMiddleware     # User and project memory loading
SkillsMiddleware          # Progressive skill disclosure
ShellMiddleware           # Shell command execution
InterruptOnConfig         # Human-in-the-loop approvals
```

**Middleware Composition**: Middleware is applied in sequence, each wrapping the previous layer. Order matters for proper functionality.

**LangChain Docs**: [Deep Agents Middleware](https://docs.langchain.com/oss/python/deepagents/middleware)

### Tool System

Tools are registered with the LLM and can be invoked by the agent:

```python
# Built-in DeepAgents Tools
- write_todos           # Create/update task lists
- read_file            # Read file contents (mock FS)
- write_file           # Write file contents (mock FS)
- list_directory       # List directory contents (mock FS)
- task                 # Spawn subagent for delegation

# DeepAgents CLI Tools  
- ls                   # List real filesystem
- read_file (override) # Read real files
- write_file (override)# Write real files
- edit_file            # Targeted file edits
- glob                 # Find files by pattern
- grep                 # Search file contents
- shell / execute      # Run shell commands
- web_search           # Tavily web search
- fetch_url            # Fetch and convert URLs
- http_request         # Make HTTP requests
```

**LangChain Docs**: [Tools and Toolkits](https://docs.langchain.com/concepts/tools)

### Configuration System

DeepAgents CLI uses a settings-based configuration:

```python
# Settings Detection (deepagents_cli/config.py)
- API keys from environment variables
- Project root detection (via .git)
- Working directory tracking
- Tool availability checking

# Agent Configuration  
- Model selection (OpenAI, Anthropic, Google)
- Temperature and sampling parameters
- Recursion limits
- Verbose logging
```

### Customization Points for ChATLAS

Based on the architecture analysis, here are the key customization points:

1. **Custom Middleware**
   - `ChatLASMemoryMiddleware` - ATLAS-specific memory templates
   - `ChatLASSkillsMiddleware` - ATLAS skills (paper review, ChATLAS docs)
   - `ChatLASHITLMiddleware` - ATLAS-specific approval logic

2. **Custom Tools**
   - Existing MCP tools from ChATLAS server
   - Future: AMI query, Rucio access, Indico meetings (via local MCP server)

3. **Custom System Prompt**
   - ATLAS experiment context
   - ChATLAS documentation preferences  
   - HTCondor workflow guidance

4. **Project Detection**
   - Use `.git` for project root (same as deepagents-cli)
   - Look for `.chatlas/` directory for project memory
   - Detect ATLAS analysis repos (athena, CMake patterns)

### Relevant Links

#### LangChain Documentation
- **Deep Agents**: https://docs.langchain.com/oss/python/deepagents/overview
- **Deep Agents CLI**: https://docs.langchain.com/oss/python/deepagents/cli  
- **Deep Agents Middleware**: https://docs.langchain.com/oss/python/deepagents/middleware
- **Deep Agents Customization**: https://docs.langchain.com/oss/python/deepagents/customization
- **LangGraph**: https://docs.langchain.com/langgraph
- **LangGraph Middleware**: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#middleware
- **Human-in-the-Loop**: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop
- **LangChain Tools**: https://docs.langchain.com/concepts/tools
- **LangChain Agents**: https://docs.langchain.com/concepts/agents

#### GitHub Repositories
- **DeepAgents Monorepo**: https://github.com/langchain-ai/deepagents
- **DeepAgents Core**: https://github.com/langchain-ai/deepagents/tree/master/libs/deepagents
- **DeepAgents CLI**: https://github.com/langchain-ai/deepagents/tree/master/libs/deepagents-cli
- **LangGraph**: https://github.com/langchain-ai/langgraph
- **LangChain**: https://github.com/langchain-ai/langchain

#### Model Context Protocol (MCP)
- **MCP Specification**: https://modelcontextprotocol.io/
- **LangChain MCP Adapters**: https://github.com/rectalogic/langchain-mcp (community)

### Implementation Notes for ChATLAS Integration

1. **Reuse Core Components**
   - Use `create_deep_agent()` from deepagents core
   - Adapt memory/skills middleware patterns from deepagents-cli
   - Keep HITL approval UI (arrow-key navigation)

2. **Extend with ChATLAS Features**
   - Add MCP tool integration (already present in current code)
   - Create ATLAS-specific skills (paper review, documentation search)
   - Customize memory templates for ATLAS conventions

3. **Maintain Compatibility**
   - Don't break existing `chatlas-agents run` command
   - New interactive mode should be opt-in
   - Support both configuration styles (env vars + YAML)

4. **Testing Strategy**
   - Unit test each middleware independently
   - Integration test full interactive sessions
   - Test project detection and memory loading
   - Test skill discovery and loading

---

**Last Updated**: December 15, 2025
**Target Release**: v0.2.0 (ChATLAS Interactive Mode)
