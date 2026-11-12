"""Golden dataset loading and validation."""
import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"

INTENTS = {"policy_question", "account_query", "complaint", "refund_request", "order_change",
           "other"}
ACTIONS = {"refund", "complaint", "escalate", "cancel_order", "change_address", "none", None}
CUSTOMERS = {"C001", "C002", "C003"}
ORDER_OWNER = {"ORD-1001": "C001", "ORD-1002": "C001", "ORD-2001": "C002", "ORD-2002": "C002",
               "ORD-3001": "C003"}
REQUIRED = ("id", "customer", "message", "expect_intent", "expect_action")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return cases


def validate_case(case: dict) -> list[str]:
    problems = [f"missing field {k!r}" for k in REQUIRED if k not in case]
    if problems:
        return problems
    if case["customer"] not in CUSTOMERS:
        problems.append(f"unknown customer {case['customer']!r}")
    if case["expect_intent"] not in INTENTS:
        problems.append(f"unknown intent {case['expect_intent']!r}")
    if case["expect_action"] not in ACTIONS:
        problems.append(f"unknown action {case['expect_action']!r}")
    order = case.get("expect_order_id")
    if order and ORDER_OWNER.get(order) != case["customer"]:
        problems.append(f"{order} does not belong to {case['customer']}")
    for turn in case.get("history", []):
        if turn.get("role") not in {"user", "assistant"} or "content" not in turn:
            problems.append(f"bad history turn {turn!r}")
    return problems
