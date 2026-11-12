"""LLM eval harness: run the golden dataset through the real graph and report accuracy.

    python -m evals.run [--limit N] [--delay S] [--json PATH] [--min-intent X] [--min-action X]

Uses the real Groq model (GROQ_API_KEY from .env). The graph is halted before
`approval_gate`/`execute`, so no refund, cancellation or ticket is ever created.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from evals.dataset import load_golden
from evals.scoring import aggregate

INTERRUPT_BEFORE = ["approval_gate", "execute"]


def setup_env() -> None:
    """Point the app at throwaway DBs. Must run before `app` is imported.
    Never touches GROQ_API_KEY (it comes from .env)."""
    tmp = tempfile.mkdtemp(prefix="support-evals-")
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp}/eval.db")
    os.environ.setdefault("CHECKPOINT_DB", f"{tmp}/ckpt.db")
    os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")


def build_eval_graph():
    from langgraph.checkpoint.memory import MemorySaver

    from app.db.seed import seed
    from app.db.session import init_db
    from app.graph.builder import build_graph

    init_db()
    seed()
    return build_graph(MemorySaver(), interrupt_before=INTERRUPT_BEFORE)


def run_case(graph, case: dict) -> dict:
    messages = [(t["role"], t["content"]) for t in case.get("history", [])]
    messages.append(("user", case["message"]))
    cfg = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"}}
    res = {"id": case["id"], "intent": None, "action": None, "order_id": None,
           "errored": False, "error": None}
    try:
        graph.invoke({"messages": messages, "customer_id": case["customer"]}, cfg)
        state = graph.get_state(cfg).values
    except Exception as e:  # a failed case must not abort the whole run
        res.update(errored=True, error=f"{type(e).__name__}: {e}")
        return res
    plan = state.get("proposed_action")
    res["intent"] = state.get("intent")
    res["action"] = plan.get("action") if plan else None
    res["order_id"] = plan.get("order_id") if plan else None
    if state.get("errors"):
        res.update(errored=True, error="; ".join(state["errors"]))
    return res


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def format_report(summary: dict) -> str:
    out = [
        f"Cases: {summary['total']}   Errors: {summary['errors']}",
        f"Routing  (intent) accuracy: {_pct(summary['intent_accuracy'])}",
        f"Planning (action) accuracy: {_pct(summary['action_accuracy'])}",
        "",
        f"{'intent':<18}{'n':>4}{'routing':>10}{'planning':>10}",
    ]
    for intent, v in sorted(summary["per_intent"].items()):
        out.append(f"{intent:<18}{v['n']:>4}{_pct(v['intent_accuracy']):>10}"
                   f"{_pct(v['action_accuracy']):>10}")
    out += ["", "Confusion (expected -> got):"]
    for exp, row in sorted(summary["confusion"].items()):
        out.append(f"  {exp}: " + ", ".join(f"{g}={n}" for g, n in sorted(row.items())))
    if summary["failures"]:
        out += ["", f"Failures ({len(summary['failures'])}):"]
        for f in summary["failures"]:
            out.append(f"  [{f['id']}] {f['message']!r}")
            out.append(f"      intent {f['expect_intent']} -> {f['got_intent']}; "
                       f"action {f['expect_action']} -> {f['got_action']}")
            if f["error"]:
                out.append(f"      ERROR: {f['error']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n")[0])
    p.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between cases")
    p.add_argument("--json", dest="json_path", default=None, help="write full results here")
    p.add_argument("--min-intent", type=float, default=None)
    p.add_argument("--min-action", type=float, default=None)
    args = p.parse_args(argv)

    setup_env()
    from app.config import get_settings

    if not get_settings().groq_api_key:
        print("GROQ_API_KEY is not set (put it in .env). Aborting.", file=sys.stderr)
        return 2

    cases = load_golden()
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    graph = build_eval_graph()
    results = []
    for i, c in enumerate(cases):
        if i and args.delay > 0:
            time.sleep(args.delay)
        r = run_case(graph, c)
        results.append(r)
        flag = "ERR" if r["errored"] else "ok "
        print(f"[{i + 1}/{len(cases)}] {flag} {c['id']}: {r['intent']} / {r['action']}",
              file=sys.stderr)

    summary = aggregate(cases, results)
    print(format_report(summary))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                        encoding="utf-8")
    if args.min_intent is not None and summary["intent_accuracy"] < args.min_intent:
        return 1
    if args.min_action is not None and summary["action_accuracy"] < args.min_action:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
