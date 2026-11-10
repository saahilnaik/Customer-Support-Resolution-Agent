"""Authentication (API keys) and role-based authorization."""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from app.auth import create_api_key
from app.db.models import ApiKey
from app.db.session import add_missing_columns, session_scope
from app.main import app

ALICE = {"X-API-Key": "demo-key-alice"}
BOB = {"X-API-Key": "demo-key-bob"}
REVIEWER = {"X-API-Key": "demo-key-reviewer"}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def classify(intent, order_id=None):
    return json.dumps({"intent": intent, "priority": "medium", "order_id": order_id})


def test_missing_or_invalid_key_is_401(client):
    for method, path in [("POST", "/chat"), ("GET", "/approvals"), ("GET", "/tickets"),
                         ("GET", "/tickets/1"), ("POST", "/approvals/x")]:
        assert client.request(method, path, json={}).status_code == 401, path
        r = client.request(method, path, json={}, headers={"X-API-Key": "nope"})
        assert r.status_code == 401, path


def test_public_endpoints_need_no_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/mock/orders/ORD-1001/shipping").status_code == 200


def test_roles_are_enforced(client):
    assert client.get("/approvals", headers=ALICE).status_code == 403
    assert client.post("/approvals/x", json={"approve": True}, headers=ALICE).status_code == 403
    assert client.post("/chat", json={"message": "hi"}, headers=REVIEWER).status_code == 403


def test_customer_id_comes_from_the_key(client, script_llm):
    # impersonation attempt: Alice's key, Bob's id
    r = client.post("/chat", json={"customer_id": "C002", "message": "hi"}, headers=ALICE)
    assert r.status_code == 403
    script_llm(classify("other"), "Hello Alice!")
    r = client.post("/chat", json={"message": "hi"}, headers=ALICE)
    assert r.status_code == 200 and r.json()["reply"] == "Hello Alice!"
    # Bob can't continue Alice's thread
    tid = r.json()["thread_id"]
    r = client.post("/chat", json={"message": "hi", "thread_id": tid}, headers=BOB)
    assert r.status_code == 403


def test_customers_only_see_their_own_tickets(client, script_llm):
    plan = json.dumps({"action": "complaint", "order_id": "ORD-1001", "category": "service",
                       "description": "rude", "reason": "x", "message": "ok"})
    script_llm(classify("complaint", "ORD-1001"), plan)
    tid = client.post("/chat", json={"message": "complaint"}, headers=ALICE).json()["ticket_id"]

    assert client.get(f"/tickets/{tid}", headers=ALICE).status_code == 200
    assert client.get(f"/tickets/{tid}", headers=BOB).status_code == 404  # no existence leak
    assert client.get(f"/tickets/{tid}", headers=REVIEWER).status_code == 200
    assert [t["id"] for t in client.get("/tickets", headers=ALICE).json()] == [tid]
    assert client.get("/tickets", headers=BOB).json() == []
    assert client.get("/tickets?customer_id=C001", headers=BOB).status_code == 403
    assert len(client.get("/tickets?customer_id=C001", headers=REVIEWER).json()) == 1


def test_approval_decision_records_reviewer(client, script_llm, external):
    plan = json.dumps({"action": "refund", "order_id": "ORD-1001", "amount": 129.99,
                       "reason": "defective"})
    script_llm(classify("refund_request", "ORD-1001"), "facts", plan)
    tid = client.post("/chat", json={"message": "refund ORD-1001"}, headers=ALICE).json()["thread_id"]
    r = client.post(f"/approvals/{tid}", json={"approve": True, "note": "ok"}, headers=REVIEWER)
    assert r.status_code == 200 and r.json()["status"] == "completed"
    [row] = client.get("/approvals?status=approved", headers=REVIEWER).json()
    assert row["decided_by"] == "Demo Reviewer" and row["note"] == "ok"


def test_created_keys_are_hashed_and_can_be_revoked(client):
    key = create_api_key("reviewer", "Dana")
    assert client.get("/approvals", headers={"X-API-Key": key}).status_code == 200
    with session_scope() as s:
        row = s.query(ApiKey).filter(ApiKey.name == "Dana").one()
        assert key not in row.key_hash  # only the hash is stored
        row.active = False
    assert client.get("/approvals", headers={"X-API-Key": key}).status_code == 401


def test_create_api_key_validates_input():
    with pytest.raises(ValueError):
        create_api_key("admin", "x")
    with pytest.raises(ValueError):
        create_api_key("customer", "x")  # customer keys need a customer
    with pytest.raises(ValueError):
        create_api_key("customer", "x", customer_id="NOPE")
    assert create_api_key("customer", "alice-2", customer_id="C001")


def test_add_missing_columns_upgrades_old_schema(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as c:  # approvals table as it was before decided_by existed
        c.execute(text("CREATE TABLE approvals (id INTEGER PRIMARY KEY, thread_id VARCHAR, "
                       "customer_id VARCHAR, action JSON, status VARCHAR, note TEXT, "
                       "created_at DATETIME, decided_at DATETIME)"))
    add_missing_columns(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("approvals")}
    assert "decided_by" in cols
