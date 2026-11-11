"""Eval harness tests. No network: scoring is pure, the runner uses the scripted fake LLM."""
from evals.scoring import aggregate, score_case


def case(**kw):
    base = {"id": "c1", "customer": "C001", "message": "m", "expect_intent": "refund_request",
            "expect_action": "refund"}
    return {**base, **kw}


def result(**kw):
    base = {"id": "c1", "intent": "refund_request", "action": "refund", "order_id": "ORD-1001",
            "errored": False, "error": None}
    return {**base, **kw}


def test_score_case_match():
    assert score_case(case(), result()) == {"intent_ok": True, "action_ok": True}


def test_score_case_intent_wrong_action_right():
    assert score_case(case(), result(intent="complaint")) == {"intent_ok": False, "action_ok": True}


def test_score_case_null_action_matches_none_result():
    c = case(expect_intent="policy_question", expect_action=None)
    assert score_case(c, result(intent="policy_question", action=None))["action_ok"] is True
    assert score_case(c, result(intent="policy_question", action="none"))["action_ok"] is False


def test_score_case_order_id_checked_only_when_expected():
    assert score_case(case(), result(order_id="ORD-9"))["action_ok"] is True
    assert score_case(case(expect_order_id="ORD-1001"), result(order_id="ORD-9"))["action_ok"] is False
    assert score_case(case(expect_order_id="ORD-1001"), result())["action_ok"] is True


def test_score_case_errored_is_wrong_on_both():
    assert score_case(case(), result(errored=True, error="boom")) == {
        "intent_ok": False, "action_ok": False}


def test_aggregate_accuracy_confusion_and_failures():
    cases = [case(id="a"), case(id="b"),
             case(id="c", expect_intent="complaint", expect_action="complaint")]
    results = [result(id="a"), result(id="b", intent="complaint", action="escalate"),
               result(id="c", intent="complaint", action="complaint")]
    agg = aggregate(cases, results)
    assert agg["total"] == 3 and agg["errors"] == 0
    assert agg["intent_accuracy"] == 2 / 3 and agg["action_accuracy"] == 2 / 3
    assert agg["per_intent"]["refund_request"] == {
        "n": 2, "intent_accuracy": 0.5, "action_accuracy": 0.5}
    assert agg["confusion"]["refund_request"] == {"refund_request": 1, "complaint": 1}
    assert [f["id"] for f in agg["failures"]] == ["b"]
    assert agg["failures"][0]["got_action"] == "escalate"


def test_aggregate_empty_returns_zero_accuracy_without_error():
    agg = aggregate([], [])
    assert agg["total"] == 0 and agg["intent_accuracy"] == 0.0 and agg["action_accuracy"] == 0.0


def test_aggregate_all_errored_is_zero():
    agg = aggregate([case()], [result(errored=True, error="x")])
    assert agg["errors"] == 1 and agg["intent_accuracy"] == 0.0 and agg["action_accuracy"] == 0.0
    assert agg["failures"][0]["error"] == "x"


# ---- Task 3: golden dataset + loader ----
import json  # noqa: E402

import pytest  # noqa: E402

from evals.dataset import GOLDEN_PATH, load_golden, validate_case  # noqa: E402


def test_golden_file_is_valid():
    cases = load_golden()
    assert len(cases) >= 35
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert validate_case(c) == [], (c["id"], validate_case(c))
    for intent in ("policy_question", "account_query", "complaint", "refund_request",
                   "order_change", "other"):
        assert sum(c["expect_intent"] == intent for c in cases) >= 2, intent
    for action in ("refund", "complaint", "escalate", "cancel_order", "change_address", "none"):
        assert sum(c["expect_action"] == action for c in cases) >= 2, action
    assert GOLDEN_PATH.exists()


def test_load_golden_skips_blank_lines(tmp_path):
    p = tmp_path / "g.jsonl"
    good = {"id": "a", "customer": "C001", "message": "hi", "expect_intent": "other",
            "expect_action": None}
    p.write_text("\n" + json.dumps(good) + "\n\n" + json.dumps({**good, "id": "b"}) + "\n")
    assert [c["id"] for c in load_golden(p)] == ["a", "b"]


def test_load_golden_reports_line_number_on_bad_json(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "a"}\n{not json}\n')
    with pytest.raises(ValueError, match=r":2:"):
        load_golden(p)


def test_validate_case_rejects_order_of_another_customer():
    c = {"id": "x", "customer": "C002", "message": "m", "expect_intent": "order_change",
         "expect_action": "cancel_order", "expect_order_id": "ORD-3001"}
    assert any("ORD-3001" in p for p in validate_case(c))


# ---- Task 4: runner ----
import os  # noqa: E402

from evals import run as run_mod  # noqa: E402


def _j(**kw):
    return json.dumps(kw)


def _classify(intent, order_id=None):
    return _j(intent=intent, priority="medium", order_id=order_id)


def _case(**kw):
    base = {"id": "r1", "customer": "C003", "message": "cancel ORD-3001",
            "expect_intent": "order_change", "expect_action": "cancel_order"}
    return {**base, **kw}


def test_run_case_halts_before_execute_and_reads_action(script_llm):
    from app.db.models import Order
    from app.db.session import session_scope

    script_llm(_classify("order_change", "ORD-3001"),
               _j(action="cancel_order", order_id="ORD-3001", new_address=None, message=None))
    res = run_mod.run_case(run_mod.build_eval_graph(), _case())
    assert res["intent"] == "order_change" and res["action"] == "cancel_order"
    assert res["order_id"] == "ORD-3001" and res["errored"] is False
    with session_scope() as s:
        assert s.get(Order, "ORD-3001").status != "cancelled"


def test_run_case_answer_only_path_has_null_action(script_llm):
    script_llm(_classify("other"), "Hello! How can I help?")
    res = run_mod.run_case(run_mod.build_eval_graph(),
                           _case(message="hi", expect_intent="other", expect_action=None))
    assert res["intent"] == "other" and res["action"] is None and res["errored"] is False


def test_run_case_uses_history(script_llm):
    fake = script_llm(_classify("other"), "ok")
    run_mod.run_case(run_mod.build_eval_graph(), _case(
        message="and now?", expect_intent="other", expect_action=None,
        history=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]))
    assert fake.calls == 2


def test_run_case_records_graph_exception_as_error():
    class Boom:
        def invoke(self, *a, **kw):
            raise RuntimeError("rate limit exhausted")

    res = run_mod.run_case(Boom(), _case())
    assert res["errored"] is True and "rate limit exhausted" in res["error"]
    assert res["intent"] is None and res["action"] is None


def test_run_case_safe_node_fallback_is_error(script_llm):
    script_llm("this is not json")
    res = run_mod.run_case(run_mod.build_eval_graph(), _case())
    assert res["errored"] is True and res["error"]


def test_main_exits_2_without_groq_key(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "groq_api_key", "")
    assert run_mod.main(["--limit", "1"]) == 2


def test_setup_env_does_not_override_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    run_mod.setup_env()
    assert os.environ["GROQ_API_KEY"] == "abc"


def test_main_json_output_creates_parent_dir_and_threshold_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(run_mod, "build_eval_graph", lambda: object())
    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)

    def fake_run_case(graph, case):
        return {"id": case["id"], "intent": case["expect_intent"], "action": "none",
                "order_id": None, "errored": False, "error": None}

    monkeypatch.setattr(run_mod, "run_case", fake_run_case)
    out = tmp_path / "a" / "b" / "out.json"
    code = run_mod.main(["--json", str(out), "--delay", "0", "--min-action", "1.0", "--limit", "3"])
    assert code == 1
    data = json.loads(out.read_text())
    assert data["summary"]["total"] == 3 and len(data["results"]) == 3
    assert run_mod.main(["--delay", "0", "--limit", "3"]) == 0


def test_format_report_lists_failures():
    cases = [case()]
    agg = aggregate(cases, [result(intent="complaint", action="escalate")])
    text = run_mod.format_report(agg)
    assert "Routing" in text and "Planning" in text and "c1" in text
