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
