# 自然语言转 SQL Deep Agent 示例

## 概览

`text-to-sql-agent` 展示了如何用 Deep Agents 构建一个面向 Chinook SQLite 示例数据库的自然语言转 SQL Agent。用户用自然语言提出问题，Agent 通过 `SQLDatabaseToolkit` 暴露的 SQL 工具探索表结构、生成只读 SQL、执行查询，并把结果整理成可读回答。

该示例强调 Deep Agents 的几个能力：用 `write_todos` 规划复杂查询、用 `memory` 加载常驻指令、用 `skills` 按需加载 SQL 查询与 schema 探索流程，以及用 `FilesystemBackend` 提供文件系统后端。

## 适用场景

- 想把自然语言问题转换为 SQLite 查询，并返回业务化解释。
- 想演示 LangChain `SQLDatabase` 与 `SQLDatabaseToolkit` 如何接入 Deep Agents。
- 想让 Agent 在复杂分析问题中先规划，再检查 schema、编写 JOIN、聚合并执行 SQL。
- 想用 `memory` 和 `skills` 实现渐进式上下文加载，避免一次性塞入所有长指令。
- 想通过 Chinook 数字媒体商店数据库练习客户、员工、发票、专辑、曲目等实体的分析查询。

## 目录结构

```text
examples/text-to-sql-agent/
├── agent.py
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── skills/
│   ├── query-writing/
│   │   └── SKILL.md
│   └── schema-exploration/
│       └── SKILL.md
└── text-to-sql-langsmith-trace.png
```

运行前还需要按 `README.md` 下载 `chinook.db` 到示例目录；该数据库文件不随源码提交。

## 运行方式

前置条件：

- Python 3.11 或更高版本。
- `ANTHROPIC_API_KEY`。
- 可选：LangSmith 相关环境变量，用于 tracing。

安装与准备：

```bash
cd examples/text-to-sql-agent
curl -L -o chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

设置环境变量：

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

可选 LangSmith tracing：

```bash
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=text2sql-deepagent
```

命令行运行：

```bash
python agent.py "What are the top 5 best-selling artists?"
python agent.py "Which employee generated the most revenue by country?"
python agent.py "How many customers are from Canada?"
```

也可以在 Python 中调用：

```python
from agent import create_sql_deep_agent

agent = create_sql_deep_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "What are the top 5 best-selling artists?"}]
})
print(result["messages"][-1].content)
```

## 核心流程

1. `agent.py` 调用 `load_dotenv()` 加载环境变量。
2. `create_sql_deep_agent()` 以示例目录为 `base_dir`，定位同目录下的 `chinook.db`。
3. 通过 `SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)` 连接 Chinook SQLite，并让 schema 信息包含 3 行样例数据。
4. 使用 `ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)` 创建模型。
5. 使用 `SQLDatabaseToolkit(db=db, llm=model)` 获取 SQL 工具，例如列出表、查看 schema、检查查询和执行查询等工具。
6. 调用 `create_deep_agent()` 组装 Deep Agent：
   - `memory=["./AGENTS.md"]` 加载常驻 Agent 指令。
   - `skills=["./skills/"]` 注册按需加载的技能目录。
   - `tools=sql_tools` 注入 SQL 工具。
   - `subagents=[]`，该示例不配置子 Agent。
   - `backend=FilesystemBackend(root_dir=base_dir)` 使用示例目录作为文件系统后端根目录。
7. CLI 将用户问题包装为 `messages` 后调用 `agent.invoke()`。
8. Agent 按问题复杂度选择直接查询，或先用 `write_todos` 规划，再探索 schema、生成 SQL、执行查询、格式化结果。

## 关键实现点

### Chinook SQLite

示例使用 Chinook 数据库，表示一个数字媒体商店，包含 artists、albums、tracks、customers、invoices、employees 等数据。数据库文件名为 `chinook.db`，需要手动下载到 `examples/text-to-sql-agent/` 目录。

### `SQLDatabaseToolkit`

`agent.py` 用 `SQLDatabaseToolkit` 把 `SQLDatabase` 包装成可供 Agent 调用的工具集合。`README.md` 将这些工具概括为：

- `list_tables`：列出数据库表。
- `get_schema`：查看表结构。
- `query_checker`：检查 SQL。
- `execute_query`：执行 SQL。

技能文件中使用的具体工具名包括 `sql_db_list_tables`、`sql_db_schema` 和 `sql_db_query`。工具命名以实际 LangChain 版本返回为准。

### `memory`

`memory=["./AGENTS.md"]` 让 Deep Agent 始终加载 `AGENTS.md`。该文件定义了 Agent 角色、数据库信息、查询规范、安全规则和复杂问题规划方式。关键约束包括：默认限制 5 行结果、只查询相关列、不使用 `SELECT *`、执行前检查 SQL、失败后分析错误并重写，以及只允许只读查询。

### `skills`

`skills=["./skills/"]` 注册两个按需技能：

- `query-writing`：用于简单 SELECT、复杂 JOIN、聚合、子查询、报表生成和查询执行。它要求复杂查询用 `write_todos` 规划，检查各表 schema，构造 `SELECT`、`FROM/JOIN`、`WHERE`、`GROUP BY`、`ORDER BY`、`LIMIT`，并处理空结果、语法错误和超时等情况。
- `schema-exploration`：用于列出表、描述列和数据类型、识别主键/外键、映射实体关系。它指导 Agent 先用 `sql_db_list_tables` 查看全部表，再用 `sql_db_schema` 检查目标表，并解释表之间的关联。

这种设计让 Agent 在上下文中先看到技能描述，只有判断当前任务需要时才加载完整 `SKILL.md`，属于 `README.md` 所说的 progressive disclosure 模式。

### 自然语言转 SQL

用户输入自然语言问题后，Agent 会根据问题复杂度选择路径：

- 简单问题，例如 “How many customers are from Canada?”：定位相关表，读取 schema，生成 `COUNT` 查询并返回结果。
- 复杂问题，例如 “Which employee generated the most revenue by country?”：用 `write_todos` 拆解任务，识别 `Employee`、`Invoice`、`InvoiceLine`、`Customer` 等相关表，规划 JOIN 和聚合逻辑，执行查询后输出分析结果。

## 可定制项

- 模型：`agent.py` 中的 `ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)` 可替换为其他兼容模型配置；示例未明确说明其他模型的验证情况。
- 数据库：可以把 `chinook.db` 替换成其他 SQLite 数据库，但需要同步调整 `AGENTS.md` 和 `skills/` 中关于业务实体、查询规范和示例关系的说明。
- 查询规范：可在 `AGENTS.md` 中调整默认 `LIMIT`、排序偏好、只读安全规则和复杂问题规划要求。
- 技能：可在 `skills/query-writing/SKILL.md` 或 `skills/schema-exploration/SKILL.md` 中扩展查询模板、错误恢复策略或 schema 解释方式。
- LangSmith：设置 `LANGCHAIN_TRACING_V2`、`LANGSMITH_ENDPOINT`、`LANGCHAIN_API_KEY` 和 `LANGCHAIN_PROJECT` 后，可观察工具调用、规划步骤、SQL、错误重试、token 使用与成本。

## 注意事项

- `chinook.db` 不随源码直接提供，需要按 `README.md` 的 `curl` 命令下载。
- 示例依赖 `ANTHROPIC_API_KEY`；没有该环境变量时无法正常调用 Anthropic 模型。
- `AGENTS.md` 明确要求只读访问，禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`TRUNCATE`、`CREATE` 等语句。
- 默认查询结果限制为 5 行，除非用户明确要求其他数量。
- 示例代码设置 `subagents=[]`，因此虽然 Deep Agents 支持子 Agent，本示例没有实际启用子 Agent。
- `FilesystemBackend(root_dir=base_dir)` 会把文件系统后端根目录设为示例目录；涉及写文件能力时应注意不要覆盖示例源码或数据库文件。
- `README.md` 中提到 `.env.example`，但当前示例目录未看到该文件；环境变量可直接在 shell 或自行创建的 `.env` 中设置。
- 该示例未明确说明自动测试命令或离线运行方式。

## 参考源码

- `examples/text-to-sql-agent/README.md`
- `examples/text-to-sql-agent/agent.py`
- `examples/text-to-sql-agent/AGENTS.md`
- `examples/text-to-sql-agent/skills/query-writing/SKILL.md`
- `examples/text-to-sql-agent/skills/schema-exploration/SKILL.md`
- `examples/text-to-sql-agent/pyproject.toml`
