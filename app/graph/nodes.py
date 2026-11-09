import json
import logging

from langchain_core.messages import SystemMessage
from langgraph.types import interrupt

from app import llm as llm_mod
from app.config import get_settings
from app.graph import common
from app.graph.common import history, last_user_text, parse_json, reply_update, safe_node
from app.rag.retriever import get_retriever
from app.tools import support_tools as tools

log = logging.getLogger(__name__)

INTENTS = {"policy_question", "account_query", "complaint", "refund_request", "other"}

SUPERVISOR_PROMPT = """You are the triage supervisor of a customer support system.
Classify the customer's latest message. Reply with ONLY a JSON object:
{"intent": one of ["policy_question","account_query","complaint","refund_request","other"],
 "priority": one of ["low","medium","high"],
 "order_id": order number mentioned such as "ORD-1001" or null}
- policy_question: general questions about policies, shipping, warranty, how things work
- account_query: questions about the customer's own orders/profile/delivery status
- complaint: reporting a problem, defect, bad experience, or asking to speak to a human
- refund_request: explicitly asking for money back / refund
- other: greetings, thanks, anything else
Use conversation history to resolve references like "that order"."""


@safe_node
def supervisor(state):
    msgs = history(state)
    out = llm_mod.invoke_llm(llm_mod.get_llm(), [SystemMessage(content=SUPERVISOR_PROMPT), *msgs])
    data = parse_json(out.content)
    intent = data.get("intent")
    if intent not in INTENTS:
        intent = "other"
    priority = data.get("priority")
    # per-turn reset of transient fields
    return {
        "intent": intent,
        "priority": priority if priority in {"low", "medium", "high"} else "medium",
        "order_id": data.get("order_id"),
        "errors": [],
        "proposed_action": None,
        "approval_status": None,
        "approval_note": None,
        "retrieved_docs": [],
        "account_summary": "",
        "ticket_id": None,
        "reply": "",
    }


KNOWLEDGE_PROMPT = """You are a support assistant answering from the company knowledge base only.
Use ONLY the context below. Cite sources like [refund_policy.md]. If the context does not
answer the question, say you don't know and offer to connect the customer with a human agent.

Context:
{context}"""


@safe_node
def knowledge(state):
    docs = get_retriever().search(last_user_text(state))
    if not docs:
        return reply_update(
            "I couldn't find that in our policies. I can escalate this to a human agent if you'd like.",
            retrieved_docs=[],
        )
    context = "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in docs)
    out = llm_mod.invoke_llm(
        llm_mod.get_llm(),
        [SystemMessage(content=KNOWLEDGE_PROMPT.format(context=context)), *history(state)],
    )
    return reply_update(str(out.content), retrieved_docs=docs)


ACCOUNT_PROMPT = """You are an account support agent. Use the tools to look up the customer's
profile and orders and answer accurately. Never invent order data. If a tool returns an error,
say so plainly. Customer id: {cid}. Be concise."""


@safe_node
def account(state):
    cid = state["customer_id"]
    text = common.run_tool_loop(
        ACCOUNT_PROMPT.format(cid=cid), history(state), tools.make_account_tools(cid)
    )
    if state.get("intent") == "refund_request":
        # facts feed the resolution planner; no reply yet
        return {"account_summary": text}
    return reply_update(text, account_summary=text)


RESOLUTION_PROMPT = """You are a resolution planner for customer support. Decide ONE action.
Reply with ONLY a JSON object:
{{"action": "refund" | "complaint" | "escalate" | "none",
  "order_id": string or null,
  "amount": number (refund only),
  "category": short string (complaint only, e.g. "defective_product","late_delivery","service"),
  "description": one-sentence summary of the issue,
  "reason": why this action,
  "message": short, friendly message to the customer}}
Rules:
- Refund only if policy allows (see policy). Never refund more than the order total.
- IMPORTANT: do NOT escalate because a refund is large or needs manager approval. The system
  routes large refunds to a manager automatically. Propose the "refund" action with the full amount.
- "complaint" registers a ticket for problems that are not refund-eligible or need follow-up.
- "escalate" if the customer asks for a human or the situation is out of policy / unclear.
- "none" if you only need to answer or need more info (put the question in "message").
Customer id: {cid}. Priority: {priority}.
Account facts:
{facts}
Relevant policy:
{policy}"""


@safe_node
def resolution_plan(state):
    policy_docs = get_retriever().search(last_user_text(state) + " refund policy", k=2)
    policy = "\n\n".join(d["text"] for d in policy_docs) or "(none found)"
    facts = state.get("account_summary") or json.dumps(tools.lookup_orders(state["customer_id"]))
    prompt = RESOLUTION_PROMPT.format(
        cid=state["customer_id"],
        priority=state.get("priority", "medium"),
        facts=facts,
        policy=policy,
    )
    out = llm_mod.invoke_llm(llm_mod.get_llm(), [SystemMessage(content=prompt), *history(state)])
    plan = parse_json(out.content)
    if plan.get("action") not in {"refund", "complaint", "escalate", "none"}:
        raise ValueError(f"invalid action from planner: {plan.get('action')!r}")
    if plan["action"] == "refund":
        try:
            plan["amount"] = round(float(plan["amount"]), 2)
        except (KeyError, TypeError, ValueError):
            raise ValueError("refund plan without a valid amount")
        if not plan.get("order_id"):
            raise ValueError("refund plan without order_id")
    return {"proposed_action": plan, "retrieved_docs": policy_docs}


def needs_approval(plan: dict | None) -> bool:
    return (
        bool(plan)
        and plan.get("action") == "refund"
        and plan.get("amount", 0) > get_settings().refund_approval_threshold
    )


def approval_gate(state):
    """Pause for a human decision on risky actions. No side effects happen before
    interrupt(), so re-running this node on resume is safe."""
    plan = state.get("proposed_action")
    if not needs_approval(plan):
        return {"approval_status": "not_required"}
    decision = interrupt(
        {
            "type": "approval_required",
            "customer_id": state["customer_id"],
            "action": plan,
            "message": f"Refund of ${plan['amount']} on {plan['order_id']} exceeds the "
            f"${get_settings().refund_approval_threshold:.0f} auto-approval limit.",
        }
    )
    approved = bool(decision.get("approved"))
    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_note": decision.get("note"),
    }


def execute(state):
    """Deterministically carries out the (possibly human-approved) action."""
    cid = state["customer_id"]
    plan = state.get("proposed_action") or {"action": "none", "message": "How else can I help?"}
    action = plan["action"]
    priority = state.get("priority", "medium")

    if action == "refund" and state.get("approval_status") == "rejected":
        note = state.get("approval_note") or "not approved by a manager"
        t = tools.create_ticket(
            cid,
            "refund_denied",
            f"Refund of ${plan['amount']} on {plan['order_id']} rejected: {note}",
            plan.get("order_id"),
            priority,
            status="escalated",
        )
        return reply_update(
            f"I'm sorry - a manager reviewed your refund request and couldn't approve it ({note}). "
            f"I've opened ticket #{t['ticket_id']} so our team can follow up with you.",
            ticket_id=t["ticket_id"],
        )

    if action == "refund":
        res = tools.process_refund(
            cid, plan["order_id"], plan["amount"], plan.get("reason", "customer request")
        )
        if "error" in res:
            t = tools.create_ticket(
                cid,
                "refund_failed",
                f"Refund could not be processed: {res['error']}",
                plan.get("order_id"),
                "high",
                status="escalated",
            )
            return reply_update(
                "I wasn't able to process your refund automatically, so I've escalated it to our "
                f"team (ticket #{t['ticket_id']}). You won't need to repeat yourself.",
                ticket_id=t["ticket_id"],
                errors=[res["error"]],
            )
        return reply_update(
            f"Your refund of ${res['amount']:.2f} for order {res['order_id']} has been issued "
            f"(reference {res['refund_ref']}). It should reach your original payment method "
            "in 5-7 business days."
        )

    if action == "complaint":
        t = tools.create_ticket(
            cid,
            plan.get("category", "general"),
            plan.get("description") or last_user_text(state),
            plan.get("order_id"),
            priority,
        )
        return reply_update(
            f"I'm sorry about this. I've registered your complaint as ticket #{t['ticket_id']} "
            f"(priority: {t['priority']}). {plan.get('message', '')}".strip(),
            ticket_id=t["ticket_id"],
        )

    if action == "escalate":
        t = tools.create_ticket(
            cid,
            "escalation",
            plan.get("reason") or last_user_text(state),
            plan.get("order_id"),
            "high",
            status="escalated",
        )
        return reply_update(
            f"I've passed this to a human agent (ticket #{t['ticket_id']}); "
            "they'll reply within 1 business day.",
            ticket_id=t["ticket_id"],
        )

    return reply_update(plan.get("message") or "How can I help you today?")


GENERAL_PROMPT = (
    "You are a friendly customer support assistant. Reply briefly. Do not invent company facts."
)


@safe_node
def general(state):
    out = llm_mod.invoke_llm(
        llm_mod.get_llm(), [SystemMessage(content=GENERAL_PROMPT), *history(state)]
    )
    return reply_update(str(out.content))
