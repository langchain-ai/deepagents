"""
Human-in-the-Loop 审批服务

工作流：
  1. Agent 调用高风险工具 → LangGraph interrupt 暂停执行
  2. 应用层将待审批动作通过 HITLStore.push_pending() 写入
  3. 审批员访问 GET /decisions/{thread_id} 查看待审批列表
  4. 审批员 POST /decisions/{thread_id} 提交 approve/reject/edit 决策
  5. 应用层读取决策后调用 agent.invoke(Command(resume=decisions)) 恢复执行

部署方式：
  uvicorn deepagents.hitl.server:app --reload --port 8080
"""
from __future__ import annotations

import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ------------------------------------------------------------------
# 内存存储（生产环境替换为 Redis Pub/Sub 或数据库）
# ------------------------------------------------------------------

class HITLStore:
    """线程安全的内存审批队列，key = thread_id"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._decisions: dict[str, list[dict[str, Any]]] = {}

    def push_pending(self, thread_id: str, action: dict[str, Any]) -> None:
        with self._lock:
            self._pending.setdefault(thread_id, []).append(action)

    def get_pending(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._pending.get(thread_id, []))

    def record_decisions(
        self, thread_id: str, decisions: list[dict[str, Any]]
    ) -> None:
        with self._lock:
            self._pending[thread_id] = []
            self._decisions[thread_id] = decisions

    def pop_decisions(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._decisions.pop(thread_id, [])


# ------------------------------------------------------------------
# Pydantic 模型
# ------------------------------------------------------------------

class DecisionItem(BaseModel):
    action_id: int
    decision: Literal["approve", "reject", "edit"]
    edited_args: dict[str, Any] | None = None


class DecisionRequest(BaseModel):
    decisions: list[DecisionItem]


# ------------------------------------------------------------------
# FastAPI 工厂
# ------------------------------------------------------------------

def create_hitl_app(store: HITLStore | None = None) -> FastAPI:
    if store is None:
        store = HITLStore()

    app = FastAPI(title="DeepAgents HITL Approval API", version="1.0.0")

    @app.get("/decisions/{thread_id}")
    def get_pending(thread_id: str):
        """查看指定会话的待审批工具调用列表"""
        pending = store.get_pending(thread_id)
        return {"thread_id": thread_id, "pending": pending}

    @app.post("/decisions/{thread_id}")
    def post_decisions(thread_id: str, body: DecisionRequest):
        """提交审批决策；所有 pending 动作都必须有对应决策"""
        pending = store.get_pending(thread_id)
        if not pending:
            raise HTTPException(status_code=404, detail="No pending actions for this thread")
        if len(body.decisions) != len(pending):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(pending)} decisions, got {len(body.decisions)}",
            )
        store.record_decisions(thread_id, [d.model_dump() for d in body.decisions])
        return {"status": "resumed", "thread_id": thread_id}

    return app


# 默认单例（uvicorn 启动用）
_default_store = HITLStore()
app = create_hitl_app(_default_store)
