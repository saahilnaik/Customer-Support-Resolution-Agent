"""Heavy / adversarial coverage: unit edges, auth CLI, LLM+tool plumbing, RAG, external
client, mock service, and API-level policy/validation paths. Offline, scripted LLM."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp

from app import auth, trace
from app import llm as llm_mod
from app.db.models import ApiKey, Order, Refund, Ticket, TurnTrace
from app.db.session import session_scope
from app.graph import common
from app.integrations import external_client as ext
from app.main import app
from app.rag import retriever as rag
from app.tools import support_tools as st

KEYS = {"C001": "demo-key-alice", "C002": "demo-key-bob", "C003": "demo-key-carla"}
REVIEWER = {"X-API-Key": "demo-key-reviewer"}


def j(**kw) -> str:
    return json.dumps(kw)


def classify(intent, order_id=None, priority="medium"):
    return j(intent=intent, priority=priority, order_id=order_id)


def change(action, order_id=None, new_address=None, message=None):
    return j(action=action, order_id=order_id, new_address=new_address, message=message)


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False, headers=REVIEWER) as c:
        yield c


def chat(client, message, customer="C001", thread_id=None, **extra):
    body = {"message": message, **extra}
    if thread_id:
        body["thread_id"] = thread_id
    return client.post("/chat", json=body, headers={"X-API-Key": KEYS[customer]})


def tool_call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c-{name}"}])


# ---------------------------------------------------------------- parse_json

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go: {"a": {"b": [1, 2]}} hope that helps', {"a": {"b": [1, 2]}}),
    ('{\n "a": "multi\\nline"\n}', {"a": "multi\nline"}),
])
def test_parse_json_tolerates_fences_and_prose(text, expected):
    assert common.parse_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{not valid json}", None])
def test_parse_json_rejects_garbage(text):
    with pytest.raises(ValueError):  # JSONDecodeError is a ValueError
        common.parse_json(text)


def test_history_and_last_user_text_ignore_tool_noise():
    from langchain_core.messages import SystemMessage, ToolMessage
    msgs = [HumanMessage(content="one"), AIMessage(content="two"),
            ToolMessage(content="x", tool_call_id="1"), SystemMessage(content="sys"),
            HumanMessage(content="three")]
    state = {"messages": msgs}
    assert [m.content for m in common.history(state)] == ["one", "two", "three"]
    assert common.last_user_text(state) == "three"
    assert common.last_user_text({}) == ""
    many = {"messages": [HumanMessage(content=str(i)) for i in range(50)]}
    assert len(common.history(many)) == common.HISTORY_WINDOW


# ------------------------------------------------------------------ safe_node

def test_safe_node_converts_errors_into_escalation():
    @common.safe_node
    def boom(state):
        raise KeyError("x")

    out = boom({})
    assert out["proposed_action"]["action"] == "escalate"
    assert "boom" in out["errors"][0] and "KeyError" in out["errors"][0]


def test_safe_node_lets_graph_interrupts_through():
    @common.safe_node
    def pause(state):
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        pause({})


def test_safe_node_passes_result_through():
    @common.safe_node
    def ok(state):
        return {"reply": "hi"}

    assert ok({}) == {"reply": "hi"}


# -------------------------------------------------------------- run_tool_loop

def test_tool_loop_unknown_tool_and_raising_tool_are_fed_back(script_llm):
    @tool
    def explode() -> dict:
        """always raises"""
        raise RuntimeError("kaboom")

    fake = script_llm(tool_call("does_not_exist"), tool_call("explode"), "final answer")
    assert common.run_tool_loop("sys", [HumanMessage(content="hi")], [explode]) == "final answer"
    assert fake.calls == 3


def test_tool_loop_gives_up_after_max_steps(script_llm):
    @tool
    def noop() -> dict:
        """nothing"""
        return {}

    script_llm(*[tool_call("noop") for _ in range(5)])
    with pytest.raises(RuntimeError, match="max steps"):
        common.run_tool_loop("sys", [], [noop], max_steps=5)


# ------------------------------------------------------------------ invoke_llm

class _Transient(Exception):
    status_code = 503


def test_invoke_llm_retries_transient_then_succeeds(script_llm, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = script_llm(_Transient("a"), _Transient("b"), "ok")
    assert llm_mod.invoke_llm(fake, []).content == "ok"
    assert fake.calls == 3


def test_invoke_llm_does_not_retry_permanent_errors(script_llm, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = script_llm(ValueError("bad request"), "never reached")
    with pytest.raises(ValueError):
        llm_mod.invoke_llm(fake, [])
    assert fake.calls == 1


def test_invoke_llm_gives_up_after_configured_attempts(script_llm, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = llm_mod.get_settings().llm_retry_attempts
    fake = script_llm(*[_Transient(str(i)) for i in range(n + 3)])
    with pytest.raises(_Transient):
        llm_mod.invoke_llm(fake, [])
    assert fake.calls == n


@pytest.mark.parametrize("exc,expected", [
    (type("RateLimitError", (Exception,), {})(), True),
    (type("APIConnectionError", (Exception,), {})(), True),
    (_Transient(), True),
    (ValueError(), False),
])
def test_is_transient(exc, expected):
    assert llm_mod._is_transient(exc) is expected


def test_get_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm_mod, "get_settings", lambda: SimpleNamespace(groq_api_key=""))
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm_mod.get_llm()


# ------------------------------------------------------------------------ auth

def test_cli_create_and_revoke_key(capsys):
    auth.main(["create-key", "--role", "customer", "--customer", "C002", "--name", "Bobby"])
    key = capsys.readouterr().out.splitlines()[0]
    assert key.startswith("sk_cust_")
    with TestClient(app) as c:
        assert c.get("/tickets", headers={"X-API-Key": key}).status_code == 200
        auth.main(["revoke", "--name", "Bobby"])
        assert "revoked 1 key(s)" in capsys.readouterr().out
        assert c.get("/tickets", headers={"X-API-Key": key}).status_code == 401
        auth.main(["revoke", "--name", "Bobby"])  # idempotent
        assert "revoked 0 key(s)" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["create-key", "--role", "customer", "--name", "x"],                      # no customer
    ["create-key", "--role", "customer", "--customer", "C999", "--name", "x"],  # unknown
    ["create-key", "--role", "reviewer", "--customer", "C001", "--name", "x"],  # reviewer+cust
    ["create-key", "--role", "admin", "--name", "x"],                         # bad role
    ["revoke"],                                                               # missing name
    [],                                                                       # no command
])
def test_cli_rejects_bad_input(argv):
    with pytest.raises(SystemExit) as e:
        auth.main(argv)
    assert e.value.code == 2


def test_cli_reviewer_key_works(capsys):
    auth.main(["create-key", "--role", "reviewer", "--name", "Dana"])
    key = capsys.readouterr().out.splitlines()[0]
    with TestClient(app) as c:
        assert c.get("/approvals", headers={"X-API-Key": key}).status_code == 200


def test_keys_are_unique_and_only_hash_is_stored():
    k1 = auth.create_api_key("reviewer", "a")
    k2 = auth.create_api_key("reviewer", "a")
    assert k1 != k2
    with session_scope() as s:
        hashes = {r.key_hash for r in s.query(ApiKey)}
    assert auth.hash_key(k1) in hashes and k1 not in hashes


def test_seed_demo_keys_is_idempotent():
    auth.seed_demo_keys()
    auth.seed_demo_keys()
    with session_scope() as s:
        assert s.query(ApiKey).count() == len(auth.DEMO_KEYS)


@pytest.mark.parametrize("headers", [
    {}, {"X-API-Key": ""}, {"X-API-Key": "   "}, {"X-API-Key": "DEMO-KEY-ALICE"},
    {"X-API-Key": "demo-key-alice "}, {"Authorization": "Bearer demo-key-alice"},
])
def test_bad_or_misspelled_keys_are_401(headers):
    with TestClient(app) as c:
        for path in ("/tickets", "/approvals", "/threads/x", "/threads/x/trace"):
            assert c.get(path, headers=headers).status_code == 401, path
        assert c.post("/chat", json={"message": "hi"}, headers=headers).status_code == 401


def test_every_non_public_route_requires_a_key():
    public = {"/health", "/", "/docs", "/redoc", "/openapi.json", "/favicon.ico",
              "/docs/oauth2-redirect"}
    with TestClient(app, raise_server_exceptions=False) as c:
        for route in app.routes:
            path = getattr(route, "path", "")
            if path in public or path.startswith("/mock"):
                continue
            for method in getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}:
                url = path.replace("{thread_id}", "t").replace("{ticket_id}", "1")
                r = c.request(method, url, json={})
                assert r.status_code == 401, f"{method} {path} -> {r.status_code}"


def test_role_matrix():
    with TestClient(app) as c:
        alice = {"X-API-Key": KEYS["C001"]}
        assert c.get("/approvals", headers=alice).status_code == 403
        assert c.post("/approvals/t", json={"approve": True}, headers=alice).status_code == 403
        assert c.patch("/tickets/1", json={"status": "open"}, headers=alice).status_code == 403
        assert c.post("/chat", json={"message": "hi"}, headers=REVIEWER).status_code == 403


# --------------------------------------------------------------------- tracing

def test_trace_noop_outside_turn():
    trace.record_llm_call()
    trace.record_tool_call("x", {}, True, 1.0)
    assert trace.traced_node("n", lambda s: {"ok": 1})({}) == {"ok": 1}


def test_trace_persistence_failure_is_swallowed(monkeypatch):
    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(trace, "session_scope", broken)
    with trace.turn("t-broken", "chat") as info:
        info["status"] = "completed"  # must not raise on exit


def test_trace_turn_records_failure_and_reraises():
    with pytest.raises(ValueError):
        with trace.turn("t-fail", "chat", customer_id="C001"):
            raise ValueError("nope")
    with session_scope() as s:
        row = s.query(TurnTrace).filter(TurnTrace.thread_id == "t-fail").one()
        assert row.status == "failed" and "ValueError: nope" in row.errors[0]


# ------------------------------------------------------------------------- RAG

def test_retriever_rejects_empty_kb(tmp_path):
    with pytest.raises(RuntimeError, match="no knowledge-base"):
        rag.Retriever(kb_dir=str(tmp_path), backend="tfidf")


def test_retriever_chunks_by_heading(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\n## Cats\nCats purr loudly.\n\n## Dogs\nDogs bark at mail carriers.\n")
    chunks = rag.load_chunks(str(tmp_path))
    assert len(chunks) >= 2
    r = rag.Retriever(kb_dir=str(tmp_path), backend="tfidf")
    top = r.search("why do dogs bark at the mail carrier", k=1)
    assert top and "Dogs" in top[0]["text"] and top[0]["source"] == "a.md"


def test_retriever_fastembed_failure_falls_back_to_tfidf(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise OSError("offline")

    monkeypatch.setattr(rag, "_FastEmbedder", Boom)
    r = rag.Retriever(backend="fastembed")
    assert isinstance(r.embedder, rag._TfidfEmbedder)
    assert r.search("how long do I have to return an item for a refund")


@pytest.mark.parametrize("query,source", [
    ("how many days to request a refund", "refund_policy.md"),
    ("how long does shipping take", "shipping.md"),
    ("is my product covered by warranty", "warranty.md"),
])
def test_retriever_routes_queries_to_right_document(query, source):
    docs = rag.Retriever(backend="tfidf").search(query, k=3)
    assert docs and docs[0]["source"] == source


@pytest.mark.parametrize("query", ["", "   ", "zzzzqqqq xxyyzz", "!!!???"])
def test_retriever_returns_nothing_for_junk(query):
    assert rag.Retriever(backend="tfidf").search(query) == []


def test_retriever_respects_k():
    r = rag.Retriever(backend="tfidf")
    assert len(r.search("refund shipping warranty order", k=1)) <= 1


# ----------------------------------------------------------- external client

def _client(handler, attempts=3):
    http = httpx.Client(base_url="http://x/", transport=httpx.MockTransport(handler))
    return ext.ExternalClient(http=http, attempts=attempts, backoff=0)


def test_client_retries_transport_errors_and_429():
    seq = iter([httpx.ConnectError("refused"), httpx.Response(429), httpx.Response(200, json={"ok": 1})])

    def handler(req):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r

    assert _client(handler).get_shipping("ORD-1") == {"ok": 1}


def test_client_exhausting_retries_raises_external_error():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500)

    with pytest.raises(ext.ExternalAPIError):
        _client(handler, attempts=4).get_shipping("o")
    assert len(calls) == 4


def test_client_4xx_is_not_retried_and_exposes_status():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(409, text="locked")

    with pytest.raises(ext.ExternalAPIError) as e:
        _client(handler).update_address("o", "somewhere")
    assert len(calls) == 1 and e.value.status_code == 409


def test_client_sends_idempotency_key_and_body():
    seen = {}

    def handler(req):
        seen.update(key=req.headers.get("Idempotency-Key"), body=json.loads(req.content),
                    method=req.method, path=req.url.path)
        return httpx.Response(200, json={"refund_ref": "R"})

    _client(handler).create_refund("ORD-9", 5.5, "why", idempotency_key="k-1")
    assert seen["key"] == "k-1" and seen["body"]["amount"] == 5.5 and seen["method"] == "POST"


# ---------------------------------------------------------------- mock service

def test_mock_service_rules(client):
    assert client.get("/mock/orders/NOPE/shipping").status_code == 404
    assert client.post("/mock/orders/ORD-1001/cancel", json={}).status_code == 409   # delivered
    assert client.put("/mock/orders/ORD-1002/address", json={"address": "x"}).status_code == 409
    r = client.post("/mock/orders/ORD-1002/cancel", json={})
    assert r.status_code == 200 and r.json()["intercepted"] is True
    assert client.post("/mock/orders/ORD-1002/cancel", json={}).json() == r.json()  # idempotent
    assert client.get("/mock/orders/ORD-1002/shipping").json()["status"] == "cancelled"
    # cancelled parcel can no longer be re-addressed, even if it was label_pending
    client.post("/mock/orders/ORD-3001/cancel", json={})
    assert client.put("/mock/orders/ORD-3001/address", json={"address": "x"}).status_code == 409
    assert client.post("/mock/refunds", json={"order_id": "o", "amount": "NaNx", "reason": "r"}).status_code == 422


def test_mock_refund_idempotency_key_dedupes_and_chaos_toggles(client):
    body = {"order_id": "ORD-1", "amount": 5, "reason": "r"}
    a = client.post("/mock/refunds", json=body, headers={"Idempotency-Key": "k1"}).json()
    b = client.post("/mock/refunds", json=body, headers={"Idempotency-Key": "k1"}).json()
    c = client.post("/mock/refunds", json=body, headers={"Idempotency-Key": "k2"}).json()
    assert a == b and a["refund_ref"] != c["refund_ref"]
    try:
        client.post("/mock/_chaos", json={"fail": True})
        assert client.post("/mock/refunds", json=body).status_code == 503
        assert client.get("/mock/orders/ORD-1001/shipping").status_code == 503
    finally:
        client.post("/mock/_chaos", json={"fail": False})
    assert client.get("/mock/orders/ORD-1001/shipping").status_code == 200


# ------------------------------------------------------------------ tools / DB

def test_lookup_functions_handle_unknowns(external):
    assert "error" in st.lookup_customer("C999")
    assert st.lookup_orders("C999") == {"orders": []}
    assert "error" in st.lookup_order_status("C001", "ORD-2001")  # someone else's
    assert "error" in st.lookup_order_status("C001", "NOPE")


@pytest.mark.parametrize("amount", [0, -1, -0.01])
def test_refund_rejects_non_positive_amounts(external, amount):
    assert "error" in st.process_refund("C001", "ORD-1001", amount, "r")
    assert external["refunds"] == []


def test_refund_cannot_exceed_total_or_touch_foreign_order(external):
    assert "error" in st.process_refund("C001", "ORD-1001", 130.00, "r")
    assert "error" in st.process_refund("C001", "ORD-2002", 1.00, "r")   # Bob's order
    assert "error" in st.process_refund("C001", "ORD-0000", 1.00, "r")   # unknown
    assert external["refunds"] == []
    assert "error" not in st.process_refund("C001", "ORD-1001", 129.99, "r")  # exact total ok


def test_cumulative_partial_refunds_cap_at_total_and_use_distinct_keys(external):
    assert "error" not in st.process_refund("C001", "ORD-1001", 100.00, "r1")
    assert "error" not in st.process_refund("C001", "ORD-1001", 29.99, "r2")
    assert "error" in st.process_refund("C001", "ORD-1001", 0.01, "r3")
    assert external["idempotency_keys"] == ["refund-ORD-1001-1", "refund-ORD-1001-2"]
    with session_scope() as s:
        assert round(sum(r.amount for r in s.query(Refund)), 2) == 129.99


def test_failed_payment_leaves_no_refund_row(external):
    external["fail"] = True
    assert "error" in st.process_refund("C001", "ORD-1001", 10, "r")
    with session_scope() as s:
        assert s.query(Refund).count() == 0


def test_refund_window_boundaries_exact(external):
    with session_scope() as s:
        o = s.get(Order, "ORD-1001")
        o.placed_at = datetime.now(timezone.utc) - timedelta(days=st.REFUND_WINDOW_DAYS, hours=-1)
    assert st.check_refund_eligibility("C001", "ORD-1001")["allowed"]
    with session_scope() as s:
        s.get(Order, "ORD-1001").placed_at = datetime.now(timezone.utc) - timedelta(days=st.REFUND_WINDOW_DAYS, hours=1)
    assert not st.check_refund_eligibility("C001", "ORD-1001")["allowed"]


def test_old_but_undelivered_order_is_not_blocked_by_window():
    with session_scope() as s:
        s.get(Order, "ORD-1002").placed_at = datetime.now(timezone.utc) - timedelta(days=400)
    assert st.check_refund_eligibility("C001", "ORD-1002")["allowed"]


@pytest.mark.parametrize("cid,oid,action,allowed,approval", [
    ("C003", "ORD-3001", "cancel_order", True, False),
    ("C003", "ORD-3001", "change_address", True, False),
    ("C001", "ORD-1002", "cancel_order", True, True),
    ("C001", "ORD-1002", "change_address", False, False),
    ("C001", "ORD-1001", "cancel_order", False, False),
    ("C001", "ORD-1001", "change_address", False, False),
    ("C001", "ORD-3001", "cancel_order", False, False),   # not Alice's
    ("C001", "NOPE", "cancel_order", False, False),
    ("C001", "ORD-1001", "delete_everything", False, False),
])
def test_order_change_matrix(cid, oid, action, allowed, approval):
    r = st.check_order_change(cid, oid, action)
    assert (r["allowed"], r["requires_approval"]) == (allowed, approval)


def test_cancel_order_twice_second_is_refused(external):
    assert st.cancel_order("C003", "ORD-3001", "changed mind")["cancelled"]
    assert "error" in st.cancel_order("C003", "ORD-3001", "again")
    assert len(external["cancels"]) == 1 and len(external["refunds"]) == 1
    assert external["refunds"][0]["amount"] == 249.00


def test_cancel_order_after_partial_refund_refunds_only_remainder(external):
    st.process_refund("C003", "ORD-3001", 50.00, "partial")
    res = st.cancel_order("C003", "ORD-3001", "x")
    assert res["refund_amount"] == 199.00


def test_cancel_order_by_wrong_customer_does_nothing(external):
    assert "error" in st.cancel_order("C001", "ORD-3001", "x")
    assert external["cancels"] == []


@pytest.mark.parametrize("addr", ["", "   ", None])
def test_change_address_requires_non_blank_address(external, addr):
    assert "error" in st.change_shipping_address("C003", "ORD-3001", addr)
    assert external["addresses"] == []


def test_address_is_stripped_and_applied(external):
    r = st.change_shipping_address("C003", "ORD-3001", "  1 Main St  ")
    assert r["address"] == "1 Main St" and external["addresses"][0]["address"] == "1 Main St"


def test_llm_tools_are_bound_to_customer(external):
    tools = {t.name: t for t in st.make_account_tools("C001")}
    assert tools["get_customer_profile"].invoke({})["customer_id"] == "C001"
    ids = {o["order_id"] for o in tools["list_my_orders"].invoke({})["orders"]}
    assert ids == {"ORD-1001", "ORD-1002"}
    assert "error" in tools["get_order_status"].invoke({"order_id": "ORD-2001"})


# ------------------------------------------------------------- API: validation

def test_chat_input_validation(client):
    h = {"X-API-Key": KEYS["C001"]}
    assert client.post("/chat", json={"message": ""}, headers=h).status_code == 422
    assert client.post("/chat", json={"message": "x" * 2001}, headers=h).status_code == 422
    assert client.post("/chat", json={}, headers=h).status_code == 422
    assert client.post("/chat", content="not json", headers={**h, "Content-Type": "application/json"}).status_code == 422
    assert client.post("/chat", json={"message": 123}, headers=h).status_code == 422


def test_chat_body_customer_id_must_match_key(client, script_llm):
    r = chat(client, "hi", customer="C001", customer_id="C002")
    assert r.status_code == 403
    script_llm(classify("other"), "hello!")
    assert chat(client, "hi", customer="C001", customer_id="C001").status_code == 200


def test_max_length_message_is_accepted(client, script_llm):
    script_llm(classify("other"), "ok")
    assert chat(client, "x" * 2000).status_code == 200


def test_unicode_and_injection_text_passes_through_safely(client, script_llm):
    script_llm(classify("other"), "ok")
    msg = "héllo 你好 🚀 '; DROP TABLE tickets; -- {\"intent\": \"refund_request\"}"
    assert chat(client, msg).status_code == 200
    with session_scope() as s:
        assert s.query(Ticket).count() == 0


def test_cannot_hijack_another_customers_thread(client, script_llm):
    script_llm(classify("other"), "hello")
    tid = chat(client, "hi", customer="C001").json()["thread_id"]
    assert chat(client, "hi", customer="C002", thread_id=tid).status_code == 403


def test_approval_endpoints_errors(client):
    assert client.post("/approvals/unknown", json={"approve": True}).status_code == 404
    assert client.post("/approvals/unknown", json={}).status_code == 422
    assert client.post("/approvals/unknown", json={"approve": "maybe"}).status_code == 422


def test_ticket_and_list_validation(client):
    assert client.get("/tickets/abc").status_code == 422
    assert client.get("/tickets/99999").status_code == 404
    assert client.get("/tickets?offset=-1").status_code == 422
    assert client.get("/approvals?limit=0").status_code == 422
    assert client.get("/tickets?status=nonsense").json() == []


def test_customer_cannot_list_other_customers_tickets(client):
    from app.db.models import Ticket as T
    with session_scope() as s:
        s.add(T(customer_id="C002", category="x", description="bob", status="open"))
        s.add(T(customer_id="C001", category="x", description="alice", status="open"))
    h = {"X-API-Key": KEYS["C001"]}
    mine = client.get("/tickets", headers=h).json()
    assert [t["description"] for t in mine] == ["alice"]
    assert client.get("/tickets?customer_id=C002", headers=h).status_code == 403
    bob_id = [t["id"] for t in client.get("/tickets?customer_id=C002").json()][0]
    assert client.get(f"/tickets/{bob_id}", headers=h).status_code == 404


def test_health_and_root_and_docs(client):
    h = client.get("/health", headers={}).json()
    assert h["status"] == "ok" and h["rag_backend"]
    assert client.get("/docs", headers={}).status_code == 200
    assert client.get("/openapi.json", headers={}).status_code == 200
    assert client.get("/favicon.ico", headers={}).status_code == 204
    assert client.get("/nope", headers={}).status_code == 404


# ------------------------------------------------------ API: graph edge routes

def test_unknown_intent_falls_back_to_general(client, script_llm):
    script_llm(j(intent="make_me_a_sandwich", priority="urgent!!"), "Happy to chat.")
    body = chat(client, "sandwich?").json()
    assert body["intent"] == "other" and body["reply"] == "Happy to chat."
    t = client.get(f"/threads/{body['thread_id']}/trace").json()["turns"][0]
    assert [n["node"] for n in t["node_path"]] == ["supervisor", "general"]


def test_supervisor_json_with_prose_and_fences_is_accepted(client, script_llm):
    script_llm("Here:\n```json\n" + classify("other", priority="high") + "\n```", "hi there")
    assert chat(client, "yo").json()["reply"] == "hi there"


def test_knowledge_with_no_matching_docs_skips_llm(client, script_llm):
    fake = script_llm(classify("policy_question"))
    body = chat(client, "zzzzqqqq xxyyzz").json()
    assert "couldn't find that" in body["reply"] and fake.calls == 1


def test_multi_turn_thread_keeps_context_and_resets_per_turn_state(client, script_llm):
    script_llm(classify("other"), "first", classify("other"), "second")
    tid = chat(client, "hello").json()["thread_id"]
    second = chat(client, "again", thread_id=tid).json()
    assert second["reply"] == "second" and second["thread_id"] == tid
    turns = client.get(f"/threads/{tid}/trace").json()["turns"]
    assert [t["kind"] for t in turns] == ["chat", "chat"]


def test_planner_invalid_action_escalates(client, script_llm):
    script_llm(classify("complaint"), j(action="delete_database"))
    body = chat(client, "I am angry").json()
    assert body["ticket_id"] and "human agent" in body["reply"]
    assert client.get(f"/tickets/{body['ticket_id']}").json()["status"] == "escalated"


@pytest.mark.parametrize("plan", [
    j(action="refund", order_id="ORD-1001"),                          # no amount
    j(action="refund", order_id="ORD-1001", amount="lots"),           # non-numeric
    j(action="refund", amount=10),                                    # no order
    j(action="refund", order_id="ORD-1001", amount=None),
])
def test_malformed_refund_plans_escalate_never_execute(client, script_llm, external, plan):
    script_llm(classify("refund_request", "ORD-1001"), "facts", plan)
    body = chat(client, "refund please").json()
    assert body["status"] == "completed" and body["ticket_id"]
    assert external["refunds"] == []


@pytest.mark.parametrize("amount", [0, -5, 129.99 + 0.01, 10_000])
def test_invalid_refund_amounts_never_reach_a_reviewer(client, script_llm, external, amount):
    plan = j(action="refund", order_id="ORD-1001", amount=amount, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "facts", plan)
    body = chat(client, "refund").json()
    assert body["status"] == "completed"
    assert client.get("/approvals").json() == [] and external["refunds"] == []


def test_refund_for_other_customers_order_is_not_executed(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-2002", amount=5, reason="r")
    script_llm(classify("refund_request", "ORD-2002"), "facts", plan)
    assert chat(client, "refund bob's order", customer="C001").json()["status"] == "completed"
    assert external["refunds"] == []


def test_refund_at_threshold_is_auto_approved_above_needs_review(client, script_llm, external):
    thr = llm_mod.get_settings().refund_approval_threshold
    at = j(action="refund", order_id="ORD-1001", amount=thr, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "facts", at)
    assert chat(client, "refund").json()["status"] == "completed"
    assert len(external["refunds"]) == 1

    above = j(action="refund", order_id="ORD-1001", amount=thr + 0.01, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "facts", above)
    assert chat(client, "refund more").json()["status"] == "awaiting_approval"


def test_double_refund_beyond_total_is_blocked_on_second_request(client, script_llm, external):
    full = j(action="refund", order_id="ORD-2002", amount=24.50, reason="r")
    script_llm(classify("refund_request", "ORD-2002"), "f", full,
               classify("refund_request", "ORD-2002"), "f", full)
    assert chat(client, "refund", customer="C002").json()["status"] == "completed"
    second = chat(client, "refund again", customer="C002").json()
    assert second["status"] == "completed" and len(external["refunds"]) == 1


def test_stale_approval_rechecks_policy_at_execution(client, script_llm, external):
    """Order changes state while waiting for a reviewer: execution must re-check."""
    script_llm(classify("order_change", "ORD-1002"), change("cancel_order", "ORD-1002"))
    tid = chat(client, "cancel ORD-1002").json()["thread_id"]
    with session_scope() as s:
        s.get(Order, "ORD-1002").status = "delivered"   # arrived while waiting
    done = client.post(f"/approvals/{tid}", json={"approve": True}).json()
    assert external["cancels"] == [] and external["refunds"] == []
    assert done["ticket_id"]  # escalated, not silently dropped


def test_approval_cannot_be_decided_twice(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "f", plan)
    tid = chat(client, "refund").json()["thread_id"]
    assert client.post(f"/approvals/{tid}", json={"approve": True}).status_code == 200
    assert client.post(f"/approvals/{tid}", json={"approve": True}).status_code == 404
    assert client.post(f"/approvals/{tid}", json={"approve": False}).status_code == 404
    assert len(external["refunds"]) == 1


def test_chat_continues_after_approval_resolves(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "f", plan, classify("other"), "anything else?")
    tid = chat(client, "refund").json()["thread_id"]
    client.post(f"/approvals/{tid}", json={"approve": True})
    assert chat(client, "thanks", thread_id=tid).json()["reply"] == "anything else?"


def test_rejected_approval_with_no_note_still_opens_ticket(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "f", plan)
    tid = chat(client, "refund").json()["thread_id"]
    done = client.post(f"/approvals/{tid}", json={"approve": False}).json()
    assert done["ticket_id"] and external["refunds"] == []


def test_approvals_list_pagination_and_pending_only(client, script_llm, external):
    for _ in range(3):
        plan = j(action="refund", order_id="ORD-1001", amount=60, reason="r")
        script_llm(classify("refund_request", "ORD-1001"), "f", plan)
        chat(client, "refund")
    assert len(client.get("/approvals").json()) == 3
    assert len(client.get("/approvals?limit=2").json()) == 2
    assert len(client.get("/approvals?limit=2&offset=2").json()) == 1


# ---- order_change node edge cases

def test_order_change_none_returns_llm_message(client, script_llm):
    script_llm(classify("order_change"), change("none", message="I can only cancel or re-address."))
    assert chat(client, "upgrade my order").json()["reply"] == "I can only cancel or re-address."


def test_order_change_none_without_message_has_default(client, script_llm):
    script_llm(classify("order_change"), change("none"))
    assert chat(client, "hmm").json()["reply"]


def test_order_change_invalid_action_escalates(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"), change("burn_it_down", "ORD-3001"))
    body = chat(client, "do something", customer="C003").json()
    assert body["ticket_id"] and external["cancels"] == []


def test_address_change_without_address_asks_for_it(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"), change("change_address", "ORD-3001"))
    body = chat(client, "change address of ORD-3001", customer="C003").json()
    assert "address" in body["reply"].lower() and external["addresses"] == []


def test_order_change_uses_order_id_from_supervisor_state(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"), change("cancel_order"))  # no id in extraction
    body = chat(client, "cancel that one", customer="C003").json()
    assert "cancelled" in body["reply"] and external["cancels"] == ["ORD-3001"]


def test_order_change_on_foreign_order_is_refused_without_leak(client, script_llm, external):
    script_llm(classify("order_change", "ORD-3001"), change("cancel_order", "ORD-3001"))
    body = chat(client, "cancel ORD-3001", customer="C001").json()
    assert "couldn't find order" in body["reply"] and external["cancels"] == []


def test_address_change_after_ship_is_refused(client, script_llm, external):
    script_llm(classify("order_change", "ORD-1002"), change("change_address", "ORD-1002", "1 St"))
    body = chat(client, "move ORD-1002 to 1 St").json()
    assert "no longer" in body["reply"] and external["addresses"] == []


def test_cancel_with_external_outage_escalates(client, script_llm, external):
    external["fail"] = True
    script_llm(classify("order_change", "ORD-3001"), change("cancel_order", "ORD-3001"))
    body = chat(client, "cancel ORD-3001", customer="C003").json()
    assert body["ticket_id"]
    with session_scope() as s:
        assert s.get(Order, "ORD-3001").status == "processing"  # untouched


def test_address_change_external_failure_escalates(client, script_llm, external):
    external["fail"] = True
    script_llm(classify("order_change", "ORD-3001"), change("change_address", "ORD-3001", "9 Elm"))
    assert chat(client, "move it", customer="C003").json()["ticket_id"]


# ---- account agent via tools

def test_account_flow_cannot_read_other_customers_orders(client, script_llm, external):
    script_llm(classify("account_query"),
               tool_call("get_order_status", order_id="ORD-2001"),
               "I could not find that order.")
    body = chat(client, "status of ORD-2001", customer="C001").json()
    assert body["reply"] == "I could not find that order."
    turn = client.get(f"/threads/{body['thread_id']}/trace").json()["turns"][0]
    assert turn["tool_calls"][0]["ok"] is False


def test_account_tool_loop_runaway_escalates(client, script_llm, external):
    script_llm(classify("account_query"), *[tool_call("list_my_orders") for _ in range(6)])
    body = chat(client, "orders?").json()
    assert body["ticket_id"]


def test_llm_outage_at_supervisor_escalates(client, script_llm, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = llm_mod.get_settings().llm_retry_attempts
    script_llm(*[_Transient(str(i)) for i in range(n)])
    body = chat(client, "hi").json()
    assert body["ticket_id"] and body["status"] == "completed"


def test_complaint_flow_with_priority_and_order(client, script_llm):
    plan = j(action="complaint", order_id="ORD-1001", category="defective_product",
             description="broken", message="We'll look.")
    script_llm(classify("complaint", "ORD-1001", priority="high"), plan)
    body = chat(client, "it is broken").json()
    t = client.get(f"/tickets/{body['ticket_id']}").json()
    assert t["priority"] == "high" and t["order_id"] == "ORD-1001" and t["category"] == "defective_product"


def test_escalate_plan_creates_escalated_ticket(client, script_llm):
    script_llm(classify("complaint"), j(action="escalate", reason="wants human"))
    body = chat(client, "human please").json()
    t = client.get(f"/tickets/{body['ticket_id']}").json()
    assert t["status"] == "escalated" and t["priority"] == "high"


def test_none_plan_replies_with_message_and_creates_no_ticket(client, script_llm):
    script_llm(classify("complaint"), j(action="none", message="Which order?"))
    body = chat(client, "problem").json()
    assert body["reply"] == "Which order?" and body["ticket_id"] is None


# ---- traces & thread view

def test_trace_visibility_and_shape(client, script_llm, external):
    script_llm(classify("other"), "hello")
    tid = chat(client, "hi").json()["thread_id"]
    mine = client.get(f"/threads/{tid}/trace", headers={"X-API-Key": KEYS["C001"]})
    assert mine.status_code == 200
    turn = mine.json()["turns"][0]
    assert turn["llm_calls"] == 2 and turn["status"] == "completed"
    assert all(n["outcome"] == "ok" and n["latency_ms"] >= 0 for n in turn["node_path"])
    assert client.get(f"/threads/{tid}/trace", headers={"X-API-Key": KEYS["C002"]}).status_code == 404
    assert client.get("/threads/nope/trace").status_code == 404


def test_traced_llm_retry_attempts_are_counted(client, script_llm, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    script_llm(classify("other"), _Transient("blip"), "recovered")
    tid = chat(client, "hi").json()["thread_id"]
    assert client.get(f"/threads/{tid}/trace").json()["turns"][0]["llm_calls"] == 3


def test_thread_view_is_read_only(client, script_llm, external):
    plan = j(action="refund", order_id="ORD-1001", amount=129.99, reason="r")
    script_llm(classify("refund_request", "ORD-1001"), "f", plan)
    tid = chat(client, "refund").json()["thread_id"]
    for _ in range(3):
        client.get(f"/threads/{tid}")
    assert len(client.get("/approvals").json()) == 1
    assert external["refunds"] == []
    with session_scope() as s:
        assert s.query(TurnTrace).filter(TurnTrace.thread_id == tid).count() == 1


# ---- DB schema upgrade / seed

def test_seed_is_idempotent():
    from app.db.models import Customer
    from app.db.seed import seed
    seed()
    seed()
    with session_scope() as s:
        assert s.query(Customer).count() == 3 and s.query(Order).count() == 5


def test_init_db_is_idempotent():
    from app.db.session import init_db
    init_db()
    init_db()


# ---- sequential soak: many threads, shared state stays consistent

def test_many_sequential_conversations_soak(client, script_llm, external):
    n = 25
    script = []
    for i in range(n):
        script += [classify("complaint"), j(action="complaint", category="service",
                                            description=f"c{i}", message="ok")]
    script_llm(*script)
    ids = {chat(client, f"complaint {i}", customer=["C001", "C002", "C003"][i % 3]).json()["thread_id"]
           for i in range(n)}
    assert len(ids) == n
    assert len(client.get("/tickets?limit=200").json()) == n
    with session_scope() as s:
        assert s.query(TurnTrace).count() == n


def test_concurrent_reads_do_not_error(client):
    from concurrent.futures import ThreadPoolExecutor
    def hit(_):
        return client.get("/tickets").status_code
    with ThreadPoolExecutor(8) as ex:
        assert set(ex.map(hit, range(40))) == {200}
