"""
完整 HITL 工作流示例
演示：RedisBackend + CustomLLM + HITL 三个模块协同工作

运行前提：
  - Redis 在 localhost:6379 运行
  - 设置环境变量：CUSTOM_LLM_BASE_URL, CUSTOM_LLM_API_KEY, CUSTOM_LLM_MODEL

运行方式：
  CUSTOM_LLM_BASE_URL=https://api.deepseek.com/v1 \\
  CUSTOM_LLM_API_KEY=sk-xxx \\
  CUSTOM_LLM_MODEL=deepseek-chat \\
  python examples/hitl_agent_example.py
"""
import os
import threading
import time

import httpx
import uvicorn
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.backends import RedisBackend
from deepagents.llm import create_custom_llm
from deepagents.hitl import HITLStore, create_hitl_app


# --- 1. 自定义 LLM ---
llm = create_custom_llm(
    base_url=os.environ.get("CUSTOM_LLM_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("CUSTOM_LLM_API_KEY", "your-key-here"),
    model=os.environ.get("CUSTOM_LLM_MODEL", "deepseek-chat"),
)

# --- 2. Redis 后端 ---
backend = RedisBackend(namespace="demo-agent", ttl_seconds=3600)

# --- 3. HITL 服务 ---
hitl_store = HITLStore()
hitl_app = create_hitl_app(hitl_store)


def start_hitl_server():
    uvicorn.run(hitl_app, host="0.0.0.0", port=8080, log_level="warning")


server_thread = threading.Thread(target=start_hitl_server, daemon=True)
server_thread.start()
time.sleep(1)  # 等待服务启动


# --- 4. 敏感工具定义 ---
@tool
def write_sensitive_file(file_path: str, content: str) -> str:
    """Write content to a file (requires approval)."""
    result = backend.write(file_path, content)
    return f"Written to {result.path}" if not result.error else result.error


# --- 5. 创建带 HITL 的 Agent ---
# 注意：create_deep_agent 需要完整的 deepagents 安装（含 langgraph）
# 本示例在有完整依赖的环境下运行
try:
    from deepagents import create_deep_agent

    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=llm,
        tools=[write_sensitive_file],
        interrupt_on={
            "write_sensitive_file": {"allowed_decisions": ["approve", "reject", "edit"]}
        },
        checkpointer=checkpointer,
    )

    THREAD_ID = "demo-thread-001"
    config = {"configurable": {"thread_id": THREAD_ID}}

    # --- 6. 运行 Agent（第一阶段：触发 interrupt）---
    print("=== 启动 Agent ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Write 'Hello World' to /workspace/output.txt"}]},
        config=config,
    )

    if "__interrupt__" in result:
        pending_actions = result["__interrupt__"]
        print(f"\n Agent 暂停，等待审批 {len(pending_actions)} 个动作")
        for i, action in enumerate(pending_actions):
            print(f"  [{i}] Tool: {action.get('tool', 'unknown')}, Args: {action.get('args', {})}")

        # --- 7. 人工审批（通过 API）---
        print("\n 提交审批决策（approve）...")
        resp = httpx.post(
            f"http://localhost:8080/decisions/{THREAD_ID}",
            json={"decisions": [{"action_id": i, "decision": "approve"} for i in range(len(pending_actions))]},
        )
        print(f"审批结果: {resp.json()}")

        # --- 8. 恢复执行 ---
        from langgraph.types import Command
        final_result = agent.invoke(Command(resume={"decisions": resp.json()}), config=config)
        print("\n Agent 完成")
        if messages := final_result.get("messages"):
            print(f"最终消息: {messages[-1].content}")
    else:
        print("Agent 直接完成（未触发审批）")
        print(result)

except ImportError:
    print("完整运行需要安装 deepagents[full]，当前仅演示模块导入。")
    print("RedisBackend:", backend)
    print("HITLStore:", hitl_store)
    print("CustomLLM:", llm)
