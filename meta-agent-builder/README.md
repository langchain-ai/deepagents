# 🧠 Meta-Agent Builder

> **Automated Project Specification Generator using Deep Agents**

A sophisticated meta-agent system that automatically generates complete project specifications from a simple description.

## 🎯 What It Does

**Input:**
```
"Create a research agent with web search and document analysis"
```

**Output:**
- Complete system architecture
- Product Requirements Document (PRD)
- Technical specifications
- Implementation guides
- Code templates

All generated using a coordinated team of 7 specialist Deep Agents.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key
- Tavily API key (for web search)

### Installation

```bash
# Clone repository
cd meta-agent-builder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your API keys
```

### Run Example

```bash
# Coming soon: CLI interface
python -m meta_agent_builder generate "Your project description"
```

---

## 📁 Project Structure

```
meta-agent-builder/
├── meta_agent_builder/          # Main package
│   ├── orchestrator/            # Meta-Orchestrator
│   ├── specialists/             # 7 Specialist Agents
│   ├── backends/                # CompositeBackend config
│   ├── middleware/              # Custom middleware
│   ├── tools/                   # Custom tools
│   ├── prompts/                 # Agent prompts
│   ├── validation/              # Validation pipeline
│   └── config/                  # Configuration
│
├── tests/                       # Test suite
├── templates/                   # Project templates
├── examples/                    # Example projects
└── meta-agent-builder-specs/   # Complete documentation
```

---

## 🏗️ Architecture

### 7 Specialist Agents

1. **Documentation Specialist** - Researches Deep Agents capabilities
2. **Architecture Specialist** - Designs system architecture
3. **PRD Specialist** - Creates product requirements
4. **Context Engineering Specialist** - Designs context management
5. **Middleware Specialist** - Designs middleware stacks
6. **Orchestration Specialist** - Designs agent orchestration
7. **Implementation Specialist** - Generates implementation guides

### Key Features

- **Self-Improving Agents** - Learn and evolve between executions
- **Persistent Knowledge** - Cross-session memory using StoreBackend
- **Automated Validation** - Built-in quality checks
- **Template Reuse** - 60% faster on similar projects
- **Intelligent Orchestration** - Parallel execution where possible

---

## 🔧 Implementation Status

### ✅ Completed

- [x] Project structure
- [x] Backend configuration (CompositeBackend)
- [x] Base specialist class
- [x] Documentation tools
- [x] Architecture tools
- [x] Configuration files

### 🚧 In Progress

- [ ] Documentation Specialist implementation
- [ ] Architecture Specialist implementation
- [ ] Meta-Orchestrator

### 📋 Planned

- [ ] Remaining 5 specialists
- [ ] Custom middleware
- [ ] CLI interface
- [ ] Validation pipeline
- [ ] Test suite
- [ ] Template library

---

## 📚 Documentation

Complete technical specifications are available in `/meta-agent-builder-specs/`:

- [Technical Specification](../meta-agent-builder-specs/00-TECHNICAL_SPECIFICATION.md)
- [Implementation Guide](../meta-agent-builder-specs/implementation/IMPLEMENTATION_GUIDE.md)
- [Quick Start](../meta-agent-builder-specs/QUICK_START.md)
- [Complete Index](../meta-agent-builder-specs/INDEX.md)

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black meta_agent_builder/
ruff check meta_agent_builder/
```

### Project Configuration

- **pyproject.toml** - Project metadata and dependencies
- **requirements.txt** - Production dependencies
- **.env.template** - Environment variable template

---

## 🎓 Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | DeepAgents 0.2.9+ |
| LLM | Claude Sonnet 4.5 |
| Storage | CompositeBackend (State + Store) |
| Tools | Tavily (web search) |
| Orchestration | LangGraph |

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| First execution | < 30 min |
| Subsequent execution | < 12 min |
| Token cost reduction | 30% |
| Quality score | > 95% |

---

## 🤝 Contributing

This is an active implementation of the specifications in `/meta-agent-builder-specs/`.

To contribute:
1. Review the specifications
2. Check the implementation status above
3. Pick a pending component
4. Follow the implementation guide

---

## 📄 License

[To be determined]

---

## 🙏 Acknowledgments

- **LangChain Team** - For Deep Agents framework
- **Anthropic** - For Claude
- **Community** - For feedback and support

---

**Status:** 🚧 Active Development
**Version:** 0.1.0
**Last Updated:** 2025-11-16
