# Deep Agents - 多功能 AI Agent 框架

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
    <img alt="Deep Agents" src=".github/images/logo-dark.svg" height="40"/>
  </picture>
</p>

<h3 align="center">示例项目集</h3>

<p align="center">
  使用 Deep Agents 构建的 Agent、模式和应用程序示例。
</p>

## 📚 示例项目列表

| 示例项目 | 描述 |
|---------|------|
| [deep_research](deep_research/) | 多步骤网络研究 Agent，使用 Tavily 进行 URL 发现、并行子 Agent 和策略性反思 |
| [content-builder-agent](content-builder-agent/) | 内容写作 Agent，展示记忆（AGENTS.md）、技能和子 Agent，用于博客文章、LinkedIn 帖子和推文，并支持图片生成 |
| [text-to-sql-agent](text-to-sql-agent/) | 自然语言转 SQL Agent，具有规划、基于技能的工作流和 Chinook 演示数据库 |
| [deploy-coding-agent](deploy-coding-agent/) | `deepagents deploy` 示例：具有 LangSmith 沙箱的自主编码 Agent，用于代码执行 |
| [deploy-content-writer](deploy-content-writer/) | `deepagents deploy` 示例：具有每个用户记忆和 Supabase 认证的内容写作 Agent |
| [deploy-mcp-docs-agent](deploy-mcp-docs-agent/) | `deepagents deploy` 示例：使用 MCP 工具搜索 LangChain 文档的文档研究 Agent |
| [deploy-gtm-agent](deploy-gtm-agent/) | `deepagents deploy` 示例：协调同步和异步子 Agent 的 GTM 策略 Agent |
| [async-subagent-server](async-subagent-server/) | 自托管 Agent Protocol 服务器，将 Deep Agents 研究者公开为异步子 Agent，并带有主管 REPL |
| [nvidia_deep_agent](nvidia_deep_agent/) | 多模型 Agent，具有用于研究和 GPU 加速代码执行的 NVIDIA Nemotron Super（RAPIDS）|
| [ralph_mode](ralph_mode/) | 自主循环模式，每次迭代使用全新上下文，使用文件系统进行持久化 |
| [rlm_agent](rlm_agent/) | `create_rlm_agent` 辅助工具：使用递归 REPL + PTC 子 Agent 链包装 `create_deep_agent`，用于跨级别的并行扇出 |
| [repl_swarm](repl_swarm/) | 技能模块示例：TypeScript 编写的 `swarm` 技能在 QuickJS REPL 内部并行调度子 Agent |
| [downloading_agents](downloading_agents/) | 展示 Agent 只是一个文件夹——下载 zip、解压、运行 |
| [better-harness](better-harness/) | 使用 `better-harness` 研究成果对 Deep Agents 工具进行评估驱动的外环优化 |

每个示例都有自己详细的 README，包含设置和使用说明。

<details>
<summary><h2>贡献示例项目</h2></summary>

请参阅[贡献指南](https://docs.langchain.com/oss/python/contributing/overview)了解一般贡献指南。

添加新示例时：

- **使用 uv** 进行依赖管理，配置 `pyproject.toml` 和 `uv.lock`（提交锁文件）
- **固定 deepagents 版本** — 在依赖项中使用版本范围（例如 `>=0.3.5,<0.4.0`）
- **包含 README** — 提供清晰的设置和使用说明
- **添加测试** — 为可重用工具或复杂辅助逻辑添加测试
- **保持专注** — 每个示例应展示一个用例或工作流
- **遵循结构** — 参考现有示例的结构（参见 `deep_research/` 或 `text-to-sql-agent/` 作为参考）

</details>

## 🚀 快速开始

查看具体示例的 README 获取详细设置说明。
