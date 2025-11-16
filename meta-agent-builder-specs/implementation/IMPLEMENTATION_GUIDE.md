# IMPLEMENTATION GUIDE - Meta-Agent Builder

**Version:** 1.0
**Complexity:** Advanced
**Estimated Implementation Time:** 40-50 hours
**Prerequisites:** Python 3.10+, LangChain experience, Deep Agents knowledge

---

## 📋 TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
4. [Code Templates](#code-templates)
5. [Testing Guide](#testing-guide)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 QUICK START

### Prerequisites

```bash
# Python 3.10 or higher
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install deepagents>=0.2.9 langchain>=0.3.0 langchain-anthropic>=0.3.0 \
    langgraph>=0.2.0 tavily-python>=0.5.0 pydantic>=2.0.0 python-dotenv>=1.0.0
```

### Environment Setup

```bash
# Create .env file
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional
TAVILY_API_KEY=your_key_here
EOF
```

### Minimal Working Example

```python
# minimal_example.py
from deepagents import create_deep_agent

# Create simplest meta-orchestrator
orchestrator = create_deep_agent(
    system_prompt="You coordinate specialists to build project specs.",
    subagents=[
        {
            "name": "architecture-specialist",
            "description": "Designs system architecture",
            "system_prompt": "You design multi-agent architectures.",
            "tools": [],
        }
    ],
)

# Run
result = orchestrator.invoke({
    "messages": [{"role": "user", "content": "Design a research agent"}]
})

print(result["messages"][-1].content)
```

---

## 📁 PROJECT STRUCTURE

```
meta-agent-builder/
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
├── .env                          # Environment variables
├── .env.template                 # Template for .env
├── README.md                     # Project documentation
│
├── meta_agent_builder/           # Main package
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point
│   │
│   ├── orchestrator/            # Meta-Orchestrator
│   │   ├── __init__.py
│   │   ├── meta_orchestrator.py
│   │   └── workflows.py
│   │
│   ├── specialists/             # Specialist Agents
│   │   ├── __init__.py
│   │   ├── base.py             # Base specialist class
│   │   ├── documentation_specialist.py
│   │   ├── architecture_specialist.py
│   │   ├── prd_specialist.py
│   │   ├── context_specialist.py
│   │   ├── middleware_specialist.py
│   │   ├── orchestration_specialist.py
│   │   └── implementation_specialist.py
│   │
│   ├── backends/                # Backend Configuration
│   │   ├── __init__.py
│   │   ├── composite_config.py
│   │   └── sandbox_config.py   # Optional
│   │
│   ├── middleware/              # Custom Middleware
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── progress_tracking.py
│   │
│   ├── tools/                   # Custom Tools
│   │   ├── __init__.py
│   │   ├── documentation_tools.py
│   │   ├── architecture_tools.py
│   │   ├── validation_tools.py
│   │   └── template_tools.py
│   │
│   ├── prompts/                 # System Prompts
│   │   ├── meta_orchestrator_prompt.md
│   │   └── specialists/
│   │       ├── documentation_specialist_prompt.md
│   │       ├── architecture_specialist_prompt.md
│   │       ├── prd_specialist_prompt.md
│   │       ├── context_specialist_prompt.md
│   │       ├── middleware_specialist_prompt.md
│   │       ├── orchestration_specialist_prompt.md
│   │       └── implementation_specialist_prompt.md
│   │
│   ├── validation/              # Validation Scripts
│   │   ├── __init__.py
│   │   ├── architecture_validator.py
│   │   ├── prd_validator.py
│   │   └── integration_validator.py
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   └── diagram_utils.py
│   │
│   └── config/                  # Configuration
│       ├── __init__.py
│       ├── agents_config.yaml
│       ├── backends_config.yaml
│       └── models_config.yaml
│
├── templates/                    # Project Templates
│   ├── research_agent/
│   ├── multi_specialist_system/
│   └── orchestration_patterns/
│
├── tests/                        # Test Suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── examples/                     # Example Projects
    ├── simple_research_agent/
    └── multi_agent_system/
```

---

## 🏗️ PHASE-BY-PHASE IMPLEMENTATION

### PHASE 1: Core Infrastructure (6-8 hours)

#### Step 1.1: Project Setup

```bash
# Create project
mkdir meta-agent-builder
cd meta-agent-builder

# Initialize
git init
python -m venv venv
source venv/bin/activate

# Create structure
mkdir -p meta_agent_builder/{orchestrator,specialists,backends,middleware,tools,prompts,validation,utils,config}
touch meta_agent_builder/__init__.py
touch meta_agent_builder/__main__.py
```

#### Step 1.2: Dependencies

```toml
# pyproject.toml
[project]
name = "meta-agent-builder"
version = "0.1.0"
description = "Meta-agent system for generating Deep Agent project specifications"
requires-python = ">=3.10"
dependencies = [
    "deepagents>=0.2.9",
    "langchain>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-openai>=0.2.0",
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "tavily-python>=0.5.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
meta-agent-builder = "meta_agent_builder.__main__:main"
```

#### Step 1.3: Backend Configuration

```python
# meta_agent_builder/backends/composite_config.py

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.base import BaseStore

def create_meta_agent_backend(store: BaseStore | None = None) -> CompositeBackend:
    """Create composite backend with routing."""

    # Determine backends based on store availability
    memory_backend = StoreBackend() if store else StateBackend()
    docs_backend = StoreBackend() if store else StateBackend()
    templates_backend = StoreBackend() if store else StateBackend()

    return CompositeBackend(
        default=StateBackend(),  # Ephemeral default
        routes={
            "/memories/": memory_backend,        # Persistent knowledge
            "/docs/": docs_backend,              # Cached docs
            "/templates/": templates_backend,    # Templates
            "/project_specs/": StateBackend(),   # Current project
            "/validation/": StateBackend(),      # Validation artifacts
        }
    )
```

#### Step 1.4: Base Specialist

```python
# meta_agent_builder/specialists/base.py

from typing import Any, Callable, Sequence
from pathlib import Path
from langchain_core.tools import BaseTool
from deepagents.middleware.subagents import SubAgent

class BaseSpecialist:
    """Base class for all specialist agents."""

    def __init__(
        self,
        name: str,
        description: str,
        prompt_file: str,
        tools: Sequence[BaseTool | Callable] | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        self.name = name
        self.description = description
        self.system_prompt = self._load_prompt(prompt_file)
        self.tools = tools or []
        self.model = model

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "specialists" / filename
        return prompt_path.read_text()

    def to_subagent_config(self) -> SubAgent:
        """Convert to SubAgent configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model,
        }
```

### PHASE 2: Specialists Implementation (12-16 hours)

#### Step 2.1: Documentation Specialist

```python
# meta_agent_builder/specialists/documentation_specialist.py

from meta_agent_builder.specialists.base import BaseSpecialist
from meta_agent_builder.tools.documentation_tools import internet_search

class DocumentationSpecialist(BaseSpecialist):
    """Specialist for researching Deep Agents documentation."""

    def __init__(self):
        super().__init__(
            name="documentation-specialist",
            description=(
                "Expert in Deep Agents with persistent knowledge base. "
                "Use for Deep Agents capabilities, patterns, and best practices."
            ),
            prompt_file="documentation_specialist_prompt.md",
            tools=[internet_search],
        )
```

#### Step 2.2: Tools Implementation

```python
# meta_agent_builder/tools/documentation_tools.py

import os
from tavily import TavilyClient
from langchain_core.tools import tool

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
def internet_search(
    query: str,
    max_results: int = 5,
) -> dict:
    """Search the web for documentation and examples.

    Args:
        query: Search query
        max_results: Number of results

    Returns:
        Search results with URLs and content
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=True,
        topic="general",
    )
```

Repeat similar pattern for other specialists...

### PHASE 3: Meta-Orchestrator (8-10 hours)

```python
# meta_agent_builder/orchestrator/meta_orchestrator.py

import uuid
from pathlib import Path
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from meta_agent_builder.backends.composite_config import create_meta_agent_backend
from meta_agent_builder.specialists import create_all_specialists

class MetaOrchestrator:
    """Main orchestrator for Meta-Agent Builder."""

    def __init__(self, store=None, checkpointer=None):
        # Create backend
        self.backend = create_meta_agent_backend(store)

        # Create specialists
        self.specialists = create_all_specialists(self.backend)

        # Load system prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "meta_orchestrator_prompt.md"
        system_prompt = prompt_path.read_text()

        # Create orchestrator agent
        self.agent = create_deep_agent(
            model="claude-sonnet-4-5-20250929",
            system_prompt=system_prompt,
            subagents=[s.to_subagent_config() for s in self.specialists],
            backend=self.backend,
            checkpointer=checkpointer or MemorySaver(),
            store=store or InMemoryStore(),
        ).with_config({"recursion_limit": 1500})

    async def process_project_request(
        self,
        user_request: str,
        thread_id: str | None = None,
    ):
        """Process a project request and generate specifications.

        Args:
            user_request: User's project description
            thread_id: Optional thread ID for resumption

        Yields:
            Events from the agent execution
        """
        config = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4())
            }
        }

        async for event in self.agent.astream(
            {"messages": [{"role": "user", "content": user_request}]},
            config=config,
            stream_mode="values"
        ):
            yield event
```

### PHASE 4: CLI Implementation (4-6 hours)

```python
# meta_agent_builder/__main__.py

import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from meta_agent_builder.orchestrator.meta_orchestrator import MetaOrchestrator

console = Console()

@click.group()
def cli():
    """Meta-Agent Builder CLI"""
    pass

@cli.command()
@click.argument('project_description', type=str)
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--interactive', is_flag=True, help='Interactive mode')
async def generate(project_description, output_dir, interactive):
    """Generate project specifications."""

    # Initialize orchestrator
    orchestrator = MetaOrchestrator()

    # Process request
    console.print(f"\n🚀 [bold]Generating specifications...[/bold]\n")

    async for event in orchestrator.process_project_request(project_description):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, 'content'):
                console.print(last_msg.content)

    console.print("\n✅ [bold green]Specifications generated![/bold green]\n")

@cli.command()
def list_templates():
    """List available project templates."""
    # Implementation
    pass

def main():
    """Main entry point."""
    cli()

if __name__ == "__main__":
    main()
```

### PHASE 5: Testing (8-10 hours)

```python
# tests/integration/test_orchestrator.py

import pytest
from meta_agent_builder.orchestrator.meta_orchestrator import MetaOrchestrator

@pytest.mark.asyncio
async def test_simple_project_generation():
    """Test generating specs for a simple project."""

    orchestrator = MetaOrchestrator()

    user_request = """
    Create a research agent that:
    - Takes a query from the user
    - Searches the web
    - Compiles results into a report
    """

    results = []
    async for event in orchestrator.process_project_request(user_request):
        results.append(event)

    # Verify outputs were created
    assert len(results) > 0
    # Add more assertions
```

---

## 📝 CODE TEMPLATES

### Specialist Template

```python
# Template for creating new specialists

from meta_agent_builder.specialists.base import BaseSpecialist
from meta_agent_builder.tools import your_tools

class YourSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(
            name="your-specialist",
            description="What this specialist does",
            prompt_file="your_specialist_prompt.md",
            tools=[your_tools],
            model="claude-sonnet-4-5-20250929",
        )
```

### Tool Template

```python
# Template for creating new tools

from langchain_core.tools import tool

@tool
def your_tool(
    param1: str,
    param2: int = 5,
) -> dict:
    """Tool description for the LLM.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value
    """
    # Implementation
    result = do_something(param1, param2)
    return result
```

---

## ✅ TESTING GUIDE

### Unit Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=meta_agent_builder --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (requires API keys)
pytest tests/integration/ -v
```

### E2E Test

```bash
# Run full end-to-end test
pytest tests/e2e/test_full_workflow.py -v -s
```

---

## 🚀 DEPLOYMENT

### Local Deployment

```bash
# Install in development mode
pip install -e .

# Run
meta-agent-builder generate "Create a research agent"
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY meta_agent_builder/ ./meta_agent_builder/
COPY prompts/ ./prompts/

CMD ["python", "-m", "meta_agent_builder"]
```

---

## 🔧 TROUBLESHOOTING

### Common Issues

**Issue:** "API key not found"
**Solution:** Check .env file and ensure ANTHROPIC_API_KEY is set

**Issue:** "Module not found"
**Solution:** Run `pip install -e .` in development mode

**Issue:** "Backend not accessible"
**Solution:** Verify Store is properly initialized if using persistent backends

---

## 📚 NEXT STEPS

1. Implement all 7 specialists
2. Create comprehensive prompts
3. Add validation pipeline
4. Build template library
5. Add monitoring and logging
6. Create user documentation
7. Deploy to production

---

**Need Help?** Check the full technical specification in `/meta-agent-builder-specs/`
