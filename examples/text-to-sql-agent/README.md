# 文本到 SQL Deep Agent

这是一个由 LangChain 的 **DeepAgents** 框架驱动的自然语言转 SQL 查询智能体示例，具备规划、文件系统与子代理能力。

## 什么是 DeepAgents？

DeepAgents 是一个基于 LangGraph 的高级智能体框架，提供：

- **规划能力** - 使用 `write_todos` 工具拆解复杂任务
- **文件系统后端** - 通过文件操作保存和检索上下文
- **子代理机制** - 将特定任务委派给专用代理
- **上下文管理** - 防止复杂任务中上下文溢出

## 演示数据库

使用 [Chinook 数据库](https://github.com/lerocha/chinook-database) —— 一个代表数字媒体商店的示例数据库。

## 快速开始

### 前置条件

- Python 3.11 或更高版本
- Anthropic API key（[在此获取](https://console.anthropic.com/)）
- （可选）LangSmith API key 用于追踪（[在此注册](https://smith.langchain.com/)）

### 安装

1. 克隆 deepagents 仓库并进入根目录：
```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents
```

2. 克隆 LangChain 仓库到 `libs/langchain`（不纳入 git，便于随时拉取上游更新）：
```bash
git clone https://github.com/langchain-ai/langchain.git libs/langchain
```

3. 进入此示例目录：
```bash
cd examples/text-to-sql-agent
```

4. 下载 Chinook 数据库：
```bash
# 下载 SQLite 数据库文件
curl -L -o chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
```

5. 创建虚拟环境并安装依赖：
```bash
# 使用 uv（推荐）
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

6. 设置环境变量：
```bash
cp .env.example .env
# 编辑 .env 并填写 API keys
```

`.env` 必填：
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

可选：
```
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=text2sql-deepagent
```

## 使用方式

### 命令行

直接用自然语言问题运行：

```bash
python agent.py "What are the top 5 best-selling artists?"
```

```bash
python agent.py "Which employee generated the most revenue by country?"
```

```bash
python agent.py "How many customers are from Canada?"
```

### 代码调用

你也可以在 Python 中使用：

```python
from agent import create_sql_deep_agent

# 创建代理
agent = create_sql_deep_agent()

# 提问
result = agent.invoke({
    "messages": [{"role": "user", "content": "What are the top 5 best-selling artists?"}]
})

print(result["messages"][-1].content)
```

## DeepAgent 如何工作

### 架构

```
用户问题
     ↓
DeepAgent（带规划）
     ├─ write_todos（规划方案）
     ├─ SQL 工具
     │  ├─ list_tables
     │  ├─ get_schema
     │  ├─ query_checker
     │  └─ execute_query
     ├─ 文件系统工具（可选）
     │  ├─ ls
     │  ├─ read_file
     │  ├─ write_file
     │  └─ edit_file
     └─ 子代理（可选）
     ↓
SQLite 数据库（Chinook）
     ↓
格式化答案
```

### 配置

DeepAgent 使用 **渐进式加载** 的记忆文件与技能：

**AGENTS.md**（始终加载）包含：
- 代理身份与角色
- 核心原则与安全规则
- 通用指南
- 交流风格

**skills/**（按需加载）包含专门流程：
- **query-writing** - SQL 查询编写（简单与复杂）
- **schema-exploration** - 数据库结构与关系发现

代理会先看到技能描述，但只有当判断需要某个技能时才加载完整 SKILL.md。这个 **渐进式加载** 模式能在保持上下文高效的同时提供深度能力。

## 示例问题

### 简单问题
```
"How many customers are from Canada?"
```
代理将直接查询并返回数量。

### 需要规划的复杂问题
```
"Which employee generated the most revenue and from which countries?"
```
代理将：
1. 使用 `write_todos` 规划方案
2. 确定需要的表（Employee、Invoice、Customer）
3. 规划 JOIN 结构
4. 执行查询
5. 格式化结果并给出分析

## DeepAgent 输出示例

DeepAgent 会展示其推理过程：

```
问题：哪位员工按国家统计的收入最高？

[规划步骤]
使用 write_todos：
- [ ] 列出数据库表
- [ ] 查看 Employee 与 Invoice 的表结构
- [ ] 规划多表 JOIN 查询
- [ ] 执行查询并按员工与国家聚合
- [ ] 格式化结果

[执行步骤]
1. 列出表...
2. 获取表结构：Employee、Invoice、InvoiceLine、Customer
3. 生成 SQL 查询...
4. 执行查询...
5. 格式化结果...

[最终答案]
员工 Jane Peacock（ID: 3）产生了最高收入...
主要国家：USA（$1000）、Canada（$500）...
```

## 项目结构

```
text-to-sql-agent/
├── agent.py                      # 核心 Deep Agent 实现与 CLI
├── AGENTS.md                     # 代理身份与通用指令（始终加载）
├── skills/                       # 专项流程（按需加载）
│   ├── query-writing/
│   │   └── SKILL.md             # SQL 查询编写流程
│   └── schema-exploration/
│       └── SKILL.md             # 数据库结构发现流程
├── chinook.db                    # 示例 SQLite 数据库（下载后，已 gitignore）
├── pyproject.toml                # 项目配置与依赖
├── uv.lock                       # 依赖锁定
├── .env.example                  # 环境变量模板
├── .gitignore                    # Git 忽略规则
├── text-to-sql-langsmith-trace.png  # LangSmith Trace 示例图
└── README.md                     # 本文件
```

## 依赖说明

所有依赖都在 `pyproject.toml` 中指定。本示例刻意使用本地路径引用核心库：

- deepagents（本地）`../../libs/deepagents`
- langchain（本地）`../../libs/langchain/libs/langchain_v1`
- langchain-anthropic >= 1.3.1
- langchain-community >= 0.3.0
- langgraph >= 1.0.6
- sqlalchemy >= 2.0.0
- python-dotenv >= 1.0.0
- tavily-python >= 0.5.0
- rich >= 13.0.0

要更新 LangChain，可运行：
```bash
git -C libs/langchain pull
```

## LangSmith 集成

### 设置

1. 在 [LangSmith](https://smith.langchain.com/) 注册免费账号
2. 在账户设置中创建 API key
3. 在 `.env` 中添加：
```
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=text2sql-deepagent
```

### 你会看到什么

启用后，每次查询都会自动被追踪：

![DeepAgent LangSmith Trace Example](text-to-sql-langsmith-trace.png)

你可以查看：
- 完整执行轨迹与工具调用
- 规划步骤（write_todos）
- 文件系统操作
- Token 使用量与成本
- 生成的 SQL 查询
- 错误信息与重试

在此查看追踪结果：https://smith.langchain.com/

## 资源

- [DeepAgents 文档](https://docs.langchain.com/oss/python/deepagents/overview)
- [DeepAgents GitHub](https://github.com/langchain-ai/deepagents)
- [LangChain](https://www.langchain.com/)
- [Claude Sonnet 4.5](https://www.anthropic.com/claude)
- [Chinook 数据库](https://github.com/lerocha/chinook-database)

## 许可证

MIT

## 贡献

欢迎贡献！请随时提交 Pull Request。
