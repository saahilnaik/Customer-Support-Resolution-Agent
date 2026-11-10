"""End-to-end tests through the FastAPI app with a scripted LLM."""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app


CUSTOMER_KEYS = {"C001": "demo-key-alice", "C002": "demo-key-bob", "C003": "demo-key-carla"}


@pytest.fixture
def client():
    # default identity is the reviewer (approvals, tickets); chat() sends a customer key
    with TestClient(app, raise_server_exceptions=False,
                    headers={"X-API-Key": "demo-key-reviewer"}) as c:
        yield c


def j(**kw) -> str:
    return json.dumps(kw)


def classify(intent, order_id=None, priority="medium") -> str:
    return j(intent=intent, priority=priority, order_id=order_id)


def chat(client, message, customer="C001", thread_id=None):
    body = {"customer_id": customer, "message": message}
    if thread_id:
        body["thread_id"] = thread_id
    return client.post("/chat", json=body, headers={"X-API-Key": CUSTOMER_KEYS[customer]})


def test_policy_question_uses_rag(client, script_llm):
    script_llm(classify("policy_question"), "Refunds are allowed within 30 days [refund_policy.md].")
    r = chat(client, "What is your refund policy for returns within 30 days?")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and "30 days" in body["reply"]


def test_account_query_calls_tools(client, script_llm, external):
    tool_call = AIMessage(
        content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "ORD-1002"}, "id": "c1"}]
    )
    script_llm(classify("account_query", "ORD-1002"), tool_call, "Your order is in transit with FedEx.")
    r = chat(client, "Where is order ORD-1002?")
    assert r.json()["reply"] == "Your order is in transit with FedEx."
    assert external["shipping_calls"] == 1


def test_complaint_creates_ticket(client, script_llm):
    plan = j(action="complaint", order_id="ORD-1001", category="defective_product",
             description="Left earcup is dead", reason="defect", message="We'll follow up.")
    script_llm(classify("complaint", "ORD-1001", "high"), plan)
    body = chat(client, "My headphones from ORD-1001 are broken").json()
    assert body["ticket_id"]
    t = client.get(f"/tickets/{body['ticket_id']}").json()
    assert t["category"] == "defective_product" and t["priority"] == "high"


def test_small_refund_is_auto_approved(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-2002", amount=24.50, reason="defective")
    script_llm(classify("refund_request", "ORD-2002"), "Order ORD-2002 total $24.50, delivered.", plan)
    body = chat(client, "Refund ORD-2002 please", customer="C002").json()
    assert body["status"] == "completed" and "RF-TEST" in body["reply"]
    assert external["refunds"][0]["amount"] == 24.5


def test_large_refund_pauses_then_approval_completes(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="defective")
    script_llm(classify("refund_request", "ORD-1001"), "ORD-1001 total $129.99, delivered.", plan)
    body = chat(client, "Refund ORD-1001, it's defective").json()
    assert body["status"] == "awaiting_approval"
    tid = body["thread_id"]
    assert external["refunds"] == []  # nothing executed yet

    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["thread_id"] == tid

    # can't chat on a paused thread
    assert chat(client, "hello?", thread_id=tid).status_code == 409

    done = client.post(f"/approvals/{tid}", json={"approve": True, "note": "ok"}).json()
    assert done["status"] == "completed" and "RF-TEST" in done["reply"]
    assert len(external["refunds"]) == 1
    assert client.get("/approvals").json() == []
    assert client.post(f"/approvals/{tid}", json={"approve": True}).status_code == 404


def test_large_refund_rejected_escalates(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="defective")
    script_llm(classify("refund_request", "ORD-1001"), "facts", plan)
    tid = chat(client, "Refund ORD-1001").json()["thread_id"]
    done = client.post(f"/approvals/{tid}", json={"approve": False, "note": "outside policy"}).json()
    assert "couldn't approve" in done["reply"]
    assert external["refunds"] == []
    t = client.get(f"/tickets/{done['ticket_id']}").json()
    assert t["status"] == "escalated" and "outside policy" in t["description"]


def change(action, order_id=None, new_address=None, message=None) -> str:
    return j(action=action, order_id=order_id, new_address=new_address, message=message)


def test_cancel_processing_order_auto_completes_with_refund(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"), change("cancel_order", "ORD-3001"))
    body = chat(client, "Please cancel ORD-3001", customer="C003").json()
    assert body["status"] == "completed" and body["intent"] == "order_change"
    assert "cancelled" in body["reply"] and "RF-TEST" in body["reply"]
    assert external["cancels"] == ["ORD-3001"]
    assert external["refunds"][0]["amount"] == 249.0


def test_cancel_shipped_order_needs_approval(client, script_llm, external):
    script_llm(classify("order_change", "ORD-1002"), change("cancel_order", "ORD-1002"))
    body = chat(client, "Cancel ORD-1002").json()
    assert body["status"] == "awaiting_approval"
    assert "shipped" in body["pending_approval"]["message"]
    assert external["cancels"] == []  # nothing happens before a human decides

    done = client.post(f"/approvals/{body['thread_id']}", json={"approve": True}).json()
    assert done["status"] == "completed" and "cancelled" in done["reply"]
    assert external["cancels"] == ["ORD-1002"]


def test_rejected_cancellation_opens_escalated_ticket(client, script_llm, external):
    script_llm(classify("order_change", "ORD-1002"), change("cancel_order", "ORD-1002"))
    tid = chat(client, "Cancel ORD-1002").json()["thread_id"]
    done = client.post(f"/approvals/{tid}", json={"approve": False, "note": "already out"}).json()
    assert "couldn't approve" in done["reply"]
    assert external["cancels"] == []
    t = client.get(f"/tickets/{done['ticket_id']}").json()
    assert t["status"] == "escalated" and t["category"] == "cancel_order_denied"


def test_cancel_delivered_order_is_refused_without_side_effects(client, script_llm, external):
    script_llm(classify("order_change", "ORD-1001"), change("cancel_order", "ORD-1001"))
    body = chat(client, "Cancel ORD-1001").json()
    assert body["status"] == "completed" and "delivered" in body["reply"]
    assert body["ticket_id"] is None
    assert external["cancels"] == [] and external["refunds"] == []


def test_cancel_succeeds_but_refund_fails_escalates(client, script_llm, external):
    external["fail_refunds"] = True
    script_llm(classify("order_change", "ORD-3001"), change("cancel_order", "ORD-3001"))
    body = chat(client, "Cancel ORD-3001", customer="C003").json()
    assert "has been cancelled" in body["reply"] and f"#{body['ticket_id']}" in body["reply"]
    t = client.get(f"/tickets/{body['ticket_id']}").json()
    assert t["category"] == "refund_failed" and t["status"] == "escalated"


def test_change_address_on_processing_order(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"),
               change("change_address", "ORD-3001", "12 New St, Springfield"))
    body = chat(client, "Ship ORD-3001 to 12 New St, Springfield", customer="C003").json()
    assert body["status"] == "completed" and "12 New St" in body["reply"]
    assert external["addresses"] == [{"order_id": "ORD-3001", "address": "12 New St, Springfield"}]


def test_order_change_without_order_id_asks_which_order(client, script_llm, external):
    script_llm(classify("order_change"), change("cancel_order"))
    body = chat(client, "Cancel my order").json()
    assert body["status"] == "completed" and "which order" in body["reply"].lower()
    assert external["cancels"] == []


def test_unparseable_llm_output_escalates_instead_of_500(client, script_llm):
    script_llm("not json at all")
    r = chat(client, "hi")
    assert r.status_code == 200
    body = r.json()
    assert "human agent" in body["reply"] and body["ticket_id"]


def test_payment_provider_down_escalates(client, script_llm, external):
    external["fail"] = True
    plan = j(action="refund", order_id="ORD-2002", amount=24.50, reason="defective")
    script_llm(classify("refund_request", "ORD-2002"), "facts", plan)
    body = chat(client, "Refund ORD-2002", customer="C002").json()
    assert "escalated" in body["reply"] and body["ticket_id"]


def test_llm_outage_after_retries_escalates(client, script_llm, monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

    monkeypatch.setattr("app.config.Settings.llm_retry_max_wait", 0.0, raising=False)
    from app.config import get_settings

    get_settings().llm_retry_attempts = 2
    script_llm(RateLimitError("slow down"), RateLimitError("slow down"))
    body = chat(client, "hi").json()
    assert body["ticket_id"] and "human agent" in body["reply"]


def test_thread_ownership(client, script_llm):
    script_llm(classify("other"), "Hello!")
    tid = chat(client, "hi").json()["thread_id"]
    assert chat(client, "hi", customer="C002", thread_id=tid).status_code == 403


def test_mock_service_chaos_and_idempotent_refund(client):
    assert client.get("/mock/orders/ORD-1001/shipping").status_code == 200
    h = {"Idempotency-Key": "k1"}
    body = {"order_id": "ORD-1001", "amount": 5, "reason": "x"}
    a = client.post("/mock/refunds", json=body, headers=h).json()
    b = client.post("/mock/refunds", json=body, headers=h).json()
    assert a == b
    client.post("/mock/_chaos", json={"fail": True})
    assert client.get("/mock/orders/ORD-1001/shipping").status_code == 503
    client.post("/mock/_chaos", json={"fail": False})


def test_mock_service_order_changes(client):
    # delivered parcels can't be cancelled or re-addressed; in-transit ones are intercepted
    assert client.post("/mock/orders/ORD-2001/cancel", json={}).status_code == 409
    assert client.put("/mock/orders/ORD-2001/address", json={"address": "x"}).status_code == 409
    assert client.put("/mock/orders/ORD-3001/address", json={"address": "1 A St"}).status_code == 200
    a = client.post("/mock/orders/ORD-3001/cancel", json={}).json()
    assert a["intercepted"] is False
    b = client.post("/mock/orders/ORD-3001/cancel", json={}).json()
    assert a == b  # idempotent
    ship = client.get("/mock/orders/ORD-3001/shipping").json()
    assert ship["status"] == "cancelled" and ship["address"] == "1 A St"
    # a cancelled shipment's address is locked
    assert client.put("/mock/orders/ORD-3001/address", json={"address": "x"}).status_code == 409
