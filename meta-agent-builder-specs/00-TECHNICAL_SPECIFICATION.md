# META-AGENT BUILDER - TECHNICAL SPECIFICATION
**Version:** 1.0
**Last Updated:** 2025-11-16
**Status:** Design Complete - Ready for Implementation

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Backend Configuration](#backend-configuration)
4. [Middleware Stack](#middleware-stack)
5. [Specialist Agents](#specialist-agents)
6. [Meta-Orchestrator](#meta-orchestrator)
7. [Self-Improvement System](#self-improvement-system)
8. [Validation Pipeline](#validation-pipeline)
9. [Template Library](#template-library)
10. [Data Flows](#data-flows)
11. [Implementation Guide](#implementation-guide)
12. [Testing Strategy](#testing-strategy)
13. [Performance Metrics](#performance-metrics)
14. [Deployment](#deployment)

---

## 🎯 EXECUTIVE SUMMARY

### Purpose
Meta-Agent Builder é um sistema baseado em Deep Agents do LangChain que recebe uma descrição de projeto e automaticamente gera:
- Arquitetura completa
- Product Requirements Document (PRD)
- Especificações técnicas
- Guias de implementação
- Code templates

### Key Features
1. **Self-Improving Specialists** - Agents que aprendem e evoluem entre execuções
2. **Persistent Knowledge Base** - Conhecimento acumulado cross-session
3. **Automated Validation** - Verificação automática de qualidade
4. **Template Reuse** - Biblioteca de patterns descobertos
5. **Intelligent Parallelization** - Execução otimizada com dependency awareness
6. **Multi-Tenancy Support** - Namespace isolation por instância

### Technology Stack
- **Framework:** LangChain + LangGraph + DeepAgents 0.2.9+
- **Primary Model:** Claude Sonnet 4.5 (customizável)
- **Storage:** Composite Backend (State + Store)
- **Execution:** Optional Sandbox Backend
- **Caching:** Anthropic Prompt Caching (built-in)

### Architecture Overview
```
┌─────────────────────────────────────────────────────┐
│           META-ORCHESTRATOR AGENT                   │
│     (Coordenador + Template Matching + Learning)    │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┼────────┬─────────┬──────────┐
    ▼        ▼        ▼         ▼          ▼
┌─────┐  ┌──────┐ ┌──────┐ ┌───────┐ ┌─────────┐
│ DOC │  │ ARCH │ │ PRD  │ │ CTX   │ │ IMPL    │
│SPEC │  │SPEC  │ │SPEC  │ │SPEC   │ │ SPEC    │
└─────┘  └──────┘ └──────┘ └───────┘ └─────────┘
   │        │        │         │          │
   └────────┴────────┴─────────┴──────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
    ┌──────────────┐   ┌──────────────┐
    │ COMPOSITE    │   │ VALIDATION   │
    │ BACKEND      │   │ PIPELINE     │
    └──────────────┘   └──────────────┘
            │                 │
            ▼                 ▼
    ┌──────────────┐   ┌──────────────┐
    │  /memories/  │   │  /templates/ │
    │  /docs/      │   │  /outputs/   │
    │  (Store)     │   │  (State)     │
    └──────────────┘   └──────────────┘
```

### Performance Targets
- **First Execution:** < 30 minutes para projeto complexo
- **Subsequent Executions:** < 12 minutes (60% faster via learning)
- **Token Efficiency:** 30% reduction via prompt caching
- **Quality Score:** > 90% completeness on validation

---

## 🏗️ SYSTEM ARCHITECTURE

### Component Hierarchy

```
MetaAgentBuilder/
├── Core Layer
│   ├── MetaOrchestrator (Main coordinator)
│   └── CompositeBackend (Storage routing)
│
├── Specialist Layer
│   ├── DocumentationSpecialist
│   ├── ArchitectureSpecialist
│   ├── PRDSpecialist
│   ├── ContextEngineeringSpecialist
│   ├── MiddlewareSpecialist
│   ├── OrchestrationSpecialist
│   └── ImplementationSpecialist
│
├── Support Layer
│   ├── ValidationPipeline
│   ├── TemplateLibrary
│   └── LearningSystem
│
└── Infrastructure Layer
    ├── StateBackend (Ephemeral storage)
    ├── StoreBackend (Persistent storage)
    └── SandboxBackend (Optional execution)
```

### Technology Dependencies

```yaml
core_dependencies:
  deepagents: ">=0.2.9"
  langchain: ">=0.3.0"
  langchain-anthropic: ">=0.3.0"
  langchain-openai: ">=0.2.0"  # Optional for alternate models
  langgraph: ">=0.2.0"
  langchain-core: ">=0.3.0"

tools_dependencies:
  tavily-python: ">=0.5.0"  # For web search

utilities:
  pydantic: ">=2.0.0"
  python-dotenv: ">=1.0.0"
  pyyaml: ">=6.0"

optional_sandbox:
  modal: ">=0.63.0"  # For Modal sandbox
  runloop-api-client: ">=0.1.0"  # For Runloop sandbox
```

### Design Principles

1. **Modularity** - Each component is independent and replaceable
2. **Composability** - Specialists can be mixed and matched
3. **Persistence** - Knowledge accumulates over time
4. **Validation** - Every output is verified
5. **Optimization** - Parallel execution where possible
6. **Learning** - System improves with each execution

---

## 📁 BACKEND CONFIGURATION

### CompositeBackend Architecture

The system uses a sophisticated multi-backend strategy that routes different paths to appropriate storage backends.

#### Configuration

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

def create_meta_agent_backend(store=None):
    """Create the composite backend for Meta-Agent Builder.

    Args:
        store: LangGraph BaseStore instance for persistent storage

    Returns:
        CompositeBackend configured for the system
    """
    return CompositeBackend(
        default=StateBackend(),  # Ephemeral storage for current execution
        routes={
            # Persistent knowledge base
            "/memories/": StoreBackend() if store else StateBackend(),

            # Cached documentation
            "/docs/": StoreBackend() if store else StateBackend(),

            # Template library
            "/templates/": StoreBackend() if store else StateBackend(),

            # Current project outputs (ephemeral)
            "/project_specs/": StateBackend(),

            # Validation artifacts (ephemeral)
            "/validation/": StateBackend(),
        }
    )
```

#### Storage Zones

| Path Prefix | Backend | Purpose | Persistence |
|-------------|---------|---------|-------------|
| `/` (default) | StateBackend | Temporary files, scratch space | Thread only |
| `/memories/` | StoreBackend | Knowledge base, learnings | Cross-thread |
| `/docs/` | StoreBackend | Cached documentation | Cross-thread |
| `/templates/` | StoreBackend | Reusable patterns | Cross-thread |
| `/project_specs/` | StateBackend | Current project outputs | Thread only |
| `/validation/` | StateBackend | Validation results | Thread only |

#### Memory Structure

```
/memories/
├── orchestrator/
│   ├── successful_patterns.md
│   └── failure_learnings.md
├── documentation/
│   ├── deepagents_reference.md
│   ├── langchain_patterns.md
│   └── best_practices.md
├── architecture/
│   ├── agent_hierarchies.md
│   ├── middleware_patterns.md
│   └── backend_strategies.md
├── prd/
│   ├── requirement_templates.md
│   └── acceptance_criteria_patterns.md
└── implementation/
    ├── code_patterns.md
    └── common_pitfalls.md

/templates/
├── research_agent/
│   ├── architecture.md
│   ├── prd.md
│   └── implementation_guide.md
├── multi_specialist_system/
│   ├── architecture.md
│   ├── prd.md
│   └── implementation_guide.md
└── orchestration_patterns/
    ├── parallel_execution.md
    └── sequential_workflows.md
```

#### Sandbox Backend (Optional)

For code execution and validation:

```python
from deepagents.backends.sandbox import SandboxBackendProtocol

# Example: Modal Sandbox
from deepagents_cli.integrations.modal import ModalSandbox

sandbox = ModalSandbox()

# Composite with sandbox as default for execution
backend = CompositeBackend(
    default=sandbox,  # Supports execute()
    routes={
        "/memories/": StoreBackend(),
        # ... other routes
    }
)
```

---

## 🔧 MIDDLEWARE STACK

### Core Middleware Layers

Each agent in the system uses a carefully designed middleware stack:

```python
def create_specialist_middleware(
    specialist_name: str,
    backend: BackendProtocol,
    model: BaseChatModel,
    enable_memory: bool = True,
    enable_validation: bool = True,
) -> list[AgentMiddleware]:
    """Create middleware stack for a specialist agent."""

    middleware = []

    # Layer 1: Agent Memory (Self-Improvement)
    if enable_memory:
        middleware.append(
            AgentMemoryMiddleware(
                backend=backend,
                memory_path=f"/memories/{specialist_name}/",
            )
        )

    # Layer 2: Task Planning
    middleware.append(TodoListMiddleware())

    # Layer 3: Filesystem Access
    middleware.append(
        FilesystemMiddleware(
            backend=backend,
            system_prompt=get_filesystem_prompt(specialist_name),
        )
    )

    # Layer 4: Context Summarization (built-in)
    middleware.append(
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        )
    )

    # Layer 5: Prompt Caching (built-in, Anthropic only)
    middleware.append(
        AnthropicPromptCachingMiddleware(
            unsupported_model_behavior="ignore"
        )
    )

    # Layer 6: Tool Call Patching (built-in)
    middleware.append(PatchToolCallsMiddleware())

    # Layer 7: Validation (custom)
    if enable_validation:
        middleware.append(
            ValidationMiddleware(
                specialist_type=specialist_name,
                validation_rules=get_validation_rules(specialist_name),
            )
        )

    # Layer 8: Progress Tracking (custom)
    middleware.append(
        ProgressTrackingMiddleware(
            specialist_name=specialist_name,
        )
    )

    return middleware
```

### Custom Middleware Specifications

#### 1. AgentMemoryMiddleware

**Purpose:** Enable self-improvement by loading/saving agent instructions

**Location:** `deepagents_cli.agent_memory.AgentMemoryMiddleware` (existing)

**Configuration:**
```python
AgentMemoryMiddleware(
    backend=backend,
    memory_path="/memories/specialist_name/",
    system_prompt_template=DEFAULT_MEMORY_SNIPPET,
)
```

**Behavior:**
- Loads `/memories/{specialist}/agent.md` at startup
- Injects into system prompt via `<agent_memory>` tags
- Agent can edit its own memory file
- Changes persist across executions

#### 2. ValidationMiddleware

**Purpose:** Validate specialist outputs before completion

**Location:** `meta_agent_builder/middleware/validation.py` (NEW)

**Implementation:**
```python
class ValidationMiddleware(AgentMiddleware):
    """Validate specialist outputs against quality rules."""

    def __init__(
        self,
        specialist_type: str,
        validation_rules: dict[str, Callable],
    ):
        self.specialist_type = specialist_type
        self.validation_rules = validation_rules

    def intercept_tool_response(
        self,
        runtime: Runtime,
        tool_call: ToolCall,
        tool_result: ToolMessage,
    ) -> dict[str, Any] | None:
        """Validate outputs when written to /project_specs/"""

        if tool_call.name == "write_file":
            file_path = tool_call.args.get("file_path", "")

            if file_path.startswith("/project_specs/"):
                content = tool_call.args.get("content", "")

                # Run validation rules
                for rule_name, validator in self.validation_rules.items():
                    is_valid, message = validator(content)

                    if not is_valid:
                        # Log validation failure
                        self._log_validation_failure(
                            file_path, rule_name, message
                        )

                        # Optionally block or warn
                        # For now, just log

        return None  # Don't modify the flow
```

#### 3. ProgressTrackingMiddleware

**Purpose:** Track and report specialist progress

**Location:** `meta_agent_builder/middleware/progress_tracking.py` (NEW)

**Implementation:**
```python
class ProgressTrackingMiddleware(AgentMiddleware):
    """Track specialist progress and emit events."""

    def __init__(self, specialist_name: str):
        self.specialist_name = specialist_name
        self.start_time = None
        self.tool_calls_count = 0

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Track agent start."""
        self.start_time = time.time()
        print(f"🚀 {self.specialist_name} started")
        return None

    def intercept_model_request(
        self,
        request: ModelRequest,
        runtime: Runtime,
    ) -> ModelRequest:
        """Track each LLM call."""
        # Log or emit event
        return request

    def intercept_tool_call(
        self,
        runtime: Runtime,
        tool_call: ToolCall,
    ) -> ToolCall | None:
        """Track each tool call."""
        self.tool_calls_count += 1
        print(f"  🔧 {self.specialist_name}: {tool_call.name}")
        return None  # Don't modify

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Track agent completion."""
        elapsed = time.time() - self.start_time
        print(f"✅ {self.specialist_name} completed in {elapsed:.1f}s "
              f"({self.tool_calls_count} tools)")
        return None
```

---

See additional specification files for detailed component documentation.
