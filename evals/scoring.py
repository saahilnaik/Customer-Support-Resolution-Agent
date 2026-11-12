"""Pure scoring for the LLM eval harness (no network, no app imports)."""


def score_case(case: dict, result: dict) -> dict:
    if result.get("errored"):
        return {"intent_ok": False, "action_ok": False}
    intent_ok = result.get("intent") == case["expect_intent"]
    action_ok = result.get("action") == case["expect_action"]
    expected_order = case.get("expect_order_id")
    if action_ok and expected_order:
        action_ok = result.get("order_id") == expected_order
    return {"intent_ok": intent_ok, "action_ok": action_ok}


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def aggregate(cases: list[dict], results: list[dict]) -> dict:
    by_id = {r["id"]: r for r in results}
    scored = [(c, by_id[c["id"]], score_case(c, by_id[c["id"]])) for c in cases]
    n = len(scored)
    per_intent: dict = {}
    confusion: dict = {}
    failures = []
    for c, r, s in scored:
        bucket = per_intent.setdefault(c["expect_intent"], {"n": 0, "i": 0, "a": 0})
        bucket["n"] += 1
        bucket["i"] += s["intent_ok"]
        bucket["a"] += s["action_ok"]
        got = r.get("intent") or "(none)"
        row = confusion.setdefault(c["expect_intent"], {})
        row[got] = row.get(got, 0) + 1
        if not (s["intent_ok"] and s["action_ok"]):
            failures.append({
                "id": c["id"], "message": c["message"],
                "expect_intent": c["expect_intent"], "got_intent": r.get("intent"),
                "expect_action": c["expect_action"], "got_action": r.get("action"),
                "error": r.get("error"),
            })
    return {
        "total": n,
        "errors": sum(1 for _, r, _ in scored if r.get("errored")),
        "intent_accuracy": _ratio(sum(s["intent_ok"] for _, _, s in scored), n),
        "action_accuracy": _ratio(sum(s["action_ok"] for _, _, s in scored), n),
        "per_intent": {
            k: {"n": v["n"], "intent_accuracy": _ratio(v["i"], v["n"]),
                "action_accuracy": _ratio(v["a"], v["n"])}
            for k, v in per_intent.items()
        },
        "confusion": confusion,
        "failures": failures,
    }
