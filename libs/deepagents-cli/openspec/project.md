# Project Context

## Purpose

DeepAgents CLI 是一个开源的终端 AI 编程助手，功能类似于 Claude Code。它基于 LangChain/LangGraph 构建，提供交互式的代码生成、文件操作、shell 执行等能力。

**核心目标**：
- 提供类似 Claude Code 的终端交互体验
- 支持多种 LLM 提供商（OpenAI、Anthropic、Google）
- 可扩展的 Skills 系统，支持领域特定的能力定制
- 持久化 Agent 记忆，保留用户偏好和项目上下文

## Tech Stack

- **语言**: Python 3.11+
- **核心依赖**:
  - `deepagents`: 底层 Agent 框架（LangGraph 封装）
  - `langchain` / `langchain-openai` / `langchain-anthropic` / `langchain-google-genai`: LLM 集成
  - `prompt-toolkit`: 终端交互 UI
  - `rich`: 富文本输出和终端美化
  - `tavily-python`: 网络搜索集成
- **沙箱集成**:
  - Modal: 远程代码执行沙箱
  - Daytona: 开发环境即服务
  - Runloop: 远程执行环境

## Project Conventions

### Code Style

- 使用 `ruff` 进行代码格式化和 lint 检查
- 遵循 Google 风格的 docstring
- 行宽限制: 100 字符
- 类型注解: 使用 Python 3.11+ 的原生类型语法（如 `str | None` 而非 `Optional[str]`）
- 文件命名: 小写下划线（snake_case）

### Architecture Patterns

```
deepagents_cli/
├── main.py           # CLI 入口和主循环
├── config.py         # 配置、设置和模型创建
├── agent.py          # Agent 创建和管理
├── tools.py          # 内置工具定义（web_search, fetch_url 等）
├── execution.py      # 任务执行逻辑
├── skills/           # Skills 系统
│   ├── load.py       # Skill 加载和解析
│   ├── middleware.py # Skills 中间件
│   └── commands.py   # skills 子命令
├── integrations/     # 远程沙箱集成
│   ├── sandbox_factory.py
│   ├── modal.py
│   ├── daytona.py
│   └── runloop.py
└── ui.py            # UI 组件和输出格式化
```

**关键设计模式**：
- **Middleware 模式**: 通过 `AgentMemoryMiddleware`、`SkillsMiddleware`、`ShellMiddleware` 扩展 Agent 能力
- **Backend 抽象**: 通过 `CompositeBackend` 支持本地和远程沙箱执行
- **Provider 自动检测**: 根据模型名称模式自动识别提供商（gpt-* → openai, claude-* → anthropic）

### Testing Strategy

- 单元测试: `tests/unit_tests/`
- 集成测试: `tests/integration_tests/`
- 运行测试: `make test` 或 `uv run pytest`
- 默认测试超时: 10 秒

### Git Workflow

- 使用语义化提交信息（中英文均可）
- 提交格式: `<type>(<scope>): <description>`
  - type: feat, fix, docs, refactor, test, chore
- 功能开发先创建 OpenSpec 提案，经确认后再实现

## Domain Context

### 模型配置系统

当前模型配置通过以下机制实现：
1. **环境变量检测**: `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`
2. **模型名称自动检测**: 根据 `--model` 参数中的模式（gpt/claude/gemini）识别提供商
3. **默认模型**: 按 OpenAI → Anthropic → Google 优先级选择

**当前限制**：
- 不支持自定义 API base URL（无法接入兼容 OpenAI/Anthropic 格式的第三方服务）
- 模型名称模式硬编码，无法灵活扩展

### Skills 系统

Skills 是可复用的 Agent 能力模块：
- 存放位置: `~/.deepagents/<agent>/skills/` 或项目级 `.deepagents/skills/`
- 格式: `SKILL.md` 文件，包含 YAML frontmatter（name、description）和 Markdown 正文
- 加载方式: 按需加载（Progressive Disclosure），减少 token 消耗

### Human-in-the-Loop (HITL)

危险操作需要用户确认：
- 文件写入/编辑
- Shell 命令执行
- 网络请求
- Subagent 委派

可通过 `--auto-approve` 禁用确认提示。

## Important Constraints

1. **Python 版本**: 必须兼容 Python 3.11+
2. **依赖管理**: 使用 `uv` 进行依赖安装和虚拟环境管理
3. **向后兼容**: 环境变量和 CLI 参数的变更需保持向后兼容
4. **安全性**: 文件路径验证，防止路径遍历攻击

## External Dependencies

### LLM Providers
- **OpenAI API**: 通过 `langchain-openai` 集成
- **Anthropic API**: 通过 `langchain-anthropic` 集成
- **Google Generative AI**: 通过 `langchain-google-genai` 集成

### External Services
- **Tavily API**: 网络搜索（需要 `TAVILY_API_KEY`）
- **LangSmith**: 可选的追踪和监控（`LANGSMITH_PROJECT`）

### Sandbox Providers
- Modal (`modal>=0.65.0`)
- Daytona (`daytona>=0.113.0`)
- Runloop (`runloop-api-client>=0.69.0`)
