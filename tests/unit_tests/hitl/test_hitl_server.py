import pytest
from fastapi.testclient import TestClient
from deepagents.hitl.server import create_hitl_app, HITLStore


@pytest.fixture
def store():
    return HITLStore()


@pytest.fixture
def client(store):
    app = create_hitl_app(store)
    return TestClient(app)


def test_get_pending_returns_empty_initially(client):
    resp = client.get("/decisions/thread-001")
    assert resp.status_code == 200
    assert resp.json()["pending"] == []


def test_post_decision_stores_and_retrieves(client, store):
    # 模拟 agent 写入待审批动作
    store.push_pending("thread-001", {
        "tool": "write_file",
        "args": {"file_path": "/ws/secret.txt", "content": "data"},
    })

    resp = client.get("/decisions/thread-001")
    assert len(resp.json()["pending"]) == 1

    # 人工审批
    resp = client.post("/decisions/thread-001", json={
        "decisions": [{"action_id": 0, "decision": "approve"}]
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "resumed"


def test_reject_decision_removes_pending(client, store):
    store.push_pending("thread-002", {"tool": "delete_file", "args": {}})
    client.post("/decisions/thread-002", json={
        "decisions": [{"action_id": 0, "decision": "reject"}]
    })
    resp = client.get("/decisions/thread-002")
    assert resp.json()["pending"] == []
