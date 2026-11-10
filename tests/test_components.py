import httpx
import pytest

from app.integrations.external_client import ExternalAPIError, ExternalClient
from app.rag.retriever import Retriever
from app.tools import support_tools as t


def test_retriever_finds_refund_policy():
    r = Retriever(backend="tfidf")
    hits = r.search("am I eligible for a refund within 30 days of delivery?")
    assert hits and hits[0]["source"] == "refund_policy.md"


def test_retriever_returns_nothing_for_unrelated_query():
    assert Retriever(backend="tfidf").search("quantum chromodynamics lattice") == []


def test_tools_enforce_customer_scoping(external):
    # C002 must not see C001's order
    assert "error" in t.lookup_order_status("C002", "ORD-1001")
    ok = t.lookup_order_status("C001", "ORD-1001")
    assert ok["total"] == 129.99 and ok["shipping"]["carrier"] == "FedEx"


def test_order_status_degrades_when_external_api_down(external):
    external["fail"] = True
    res = t.lookup_order_status("C001", "ORD-1001")
    assert res["status"] == "delivered"  # DB data still returned
    assert "unavailable" in res["shipping"]["error"]


def test_refund_validation(external):
    assert "error" in t.process_refund("C001", "ORD-1001", 500.0, "x")  # > total
    assert "error" in t.process_refund("C001", "ORD-1001", -1, "x")
    ok = t.process_refund("C001", "ORD-1001", 100.0, "defective")
    assert ok["refund_ref"] == "RF-TEST"
    # second refund would exceed the order total
    assert "error" in t.process_refund("C001", "ORD-1001", 50.0, "again")


def test_order_change_eligibility_rules():
    # processing: cancel without approval, address change allowed
    c = t.check_order_change("C003", "ORD-3001", "cancel_order")
    assert c["allowed"] and not c["requires_approval"]
    assert t.check_order_change("C003", "ORD-3001", "change_address")["allowed"]
    # shipped: cancel needs a manager (carrier intercept), address is locked
    c = t.check_order_change("C001", "ORD-1002", "cancel_order")
    assert c["allowed"] and c["requires_approval"]
    assert not t.check_order_change("C001", "ORD-1002", "change_address")["allowed"]
    # delivered: neither
    c = t.check_order_change("C001", "ORD-1001", "cancel_order")
    assert not c["allowed"] and "return" in c["reason"]
    # scoping + unknown action
    assert not t.check_order_change("C002", "ORD-3001", "cancel_order")["allowed"]
    assert not t.check_order_change("C003", "ORD-3001", "teleport")["allowed"]


def test_cancel_order_refunds_remaining_and_is_not_repeatable(external):
    t.process_refund("C003", "ORD-3001", 49.0, "partial goodwill")
    res = t.cancel_order("C003", "ORD-3001", "changed my mind")
    assert res["cancelled"] and res["refund_amount"] == 200.0
    assert external["cancels"] == ["ORD-3001"]
    assert external["refunds"][-1]["amount"] == 200.0
    # each refund on the order gets its own idempotency key (retries of one refund share it)
    refund_keys = [k for k in external["idempotency_keys"] if k.startswith("refund-")]
    assert len(set(refund_keys)) == 2
    assert t.lookup_orders("C003")["orders"][0]["status"] == "cancelled"
    assert "error" in t.cancel_order("C003", "ORD-3001", "again")
    assert "error" in t.cancel_order("C001", "ORD-3001", "not mine")


def test_cancel_order_leaves_order_untouched_when_api_down(external):
    external["fail"] = True
    assert "error" in t.cancel_order("C003", "ORD-3001", "x")
    assert t.lookup_orders("C003")["orders"][0]["status"] == "processing"
    assert external["refunds"] == []


def test_change_shipping_address(external):
    ok = t.change_shipping_address("C003", "ORD-3001", "12 New St, Springfield")
    assert ok["address"] == "12 New St, Springfield"
    assert external["addresses"][0]["address"] == "12 New St, Springfield"
    assert "error" in t.change_shipping_address("C001", "ORD-1002", "x")  # shipped
    assert "error" in t.change_shipping_address("C003", "ORD-3001", "  ")  # blank


def test_client_retries_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    c = ExternalClient(http=httpx.Client(base_url="http://x/", transport=httpx.MockTransport(handler)),
                       attempts=3, backoff=0)
    assert c.get_shipping("ORD-1") == {"ok": 1}
    assert calls["n"] == 3


def test_client_gives_up_after_retries_and_does_not_retry_4xx():
    calls = {"n": 0}

    def always_503(req):
        calls["n"] += 1
        return httpx.Response(503)

    c = ExternalClient(http=httpx.Client(base_url="http://x/", transport=httpx.MockTransport(always_503)),
                       attempts=3, backoff=0)
    with pytest.raises(ExternalAPIError):
        c.get_shipping("ORD-1")
    assert calls["n"] == 3

    calls["n"] = 0

    def not_found(req):
        calls["n"] += 1
        return httpx.Response(404, text="nope")

    c = ExternalClient(http=httpx.Client(base_url="http://x/", transport=httpx.MockTransport(not_found)),
                       attempts=3, backoff=0)
    with pytest.raises(ExternalAPIError) as ei:
        c.get_shipping("ORD-1")
    assert ei.value.status_code == 404 and calls["n"] == 1
