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
