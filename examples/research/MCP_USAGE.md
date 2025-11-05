# 使用 MCP Server 的研究代理

本示例展示了如何在 DeepAgents 的研究代理中集成 MCP (Model Context Protocol) server。

## 配置说明

已添加对 `http://127.0.0.1:18060` 的 MCP server 支持，使用 SSE (Server-Sent Events) 传输协议。

## 安装依赖

```bash
cd examples/research
pip install -r requirements.txt
```

## 使用方法

### 方法 1: 使用默认代理（不含 MCP）

```python
from research_agent import agent

# 使用默认的代理（只有 internet_search 工具）
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "研究一下 LangGraph"}]},
    stream_mode="values"
):
    if "messages" in chunk:
        chunk["messages"][-1].pretty_print()
```

### 方法 2: 使用集成了 MCP 的代理

```python
import asyncio
from research_agent import create_agent_with_mcp

async def run_with_mcp():
    # 创建包含 MCP 工具的代理
    agent_with_mcp = await create_agent_with_mcp()

    # 使用代理
    async for chunk in agent_with_mcp.astream(
        {"messages": [{"role": "user", "content": "研究一下 LangGraph"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

# 运行
asyncio.run(run_with_mcp())
```

### 方法 3: 使用提供的 main 函数

取消注释 `research_agent.py` 文件末尾的代码：

```python
# 将这一行取消注释：
asyncio.run(main())
```

然后运行：

```bash
python research_agent.py
```

## MCP Server 配置

MCP server 配置位于 [research_agent.py:252-259](research_agent.py#L252-L259)：

```python
mcp_client = MultiServerMCPClient(
    {
        "httpstream": {
            "transport": "sse",
            "url": "http://127.0.0.1:18060/sse"
        }
    }
)
```

### 修改 MCP Server 地址

如果你的 MCP server 运行在不同的地址或端口，可以修改配置：

```python
mcp_client = MultiServerMCPClient(
    {
        "httpstream": {
            "transport": "sse",
            "url": "http://your-server:port/sse"
        }
    }
)
```

### 添加认证头

如果你的 MCP server 需要认证：

```python
mcp_client = MultiServerMCPClient(
    {
        "httpstream": {
            "transport": "sse",
            "url": "http://127.0.0.1:18060/sse",
            "headers": {
                "Authorization": "Bearer your-token-here"
            }
        }
    }
)
```

## 工作原理

1. **MCP 客户端初始化**: `create_agent_with_mcp()` 函数创建一个 `MultiServerMCPClient` 实例，连接到指定的 MCP server
2. **工具获取**: 通过 `await mcp_client.get_tools()` 从 MCP server 获取所有可用工具
3. **工具合并**: 将 MCP 工具与本地的 `internet_search` 工具合并
4. **代理创建**: 使用所有工具创建 DeepAgent 实例

这样，研究代理不仅可以使用 Tavily 进行网络搜索，还可以访问 MCP server 提供的所有工具。

## 环境变量

确保设置以下环境变量：

```bash
export TAVILY_API_KEY="your-tavily-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# 可选配置
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 自定义 OpenAI 端点
export OPENAI_MODEL="gpt-4o"  # 自定义模型名称
```

## 注意事项

- 确保 MCP server 在 `http://127.0.0.1:18060` 上运行并可访问
- MCP server 必须支持 SSE (Server-Sent Events) 传输协议
- 所有 MCP 工具会自动转换为 LangChain 工具格式，与本地工具无缝集成
