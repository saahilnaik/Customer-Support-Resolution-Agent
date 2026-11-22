import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import trace
from app.auth import Principal, get_principal, require_customer, require_reviewer
from app.config import get_settings
from app.db.models import Approval, Customer, Ticket, TurnTrace
from app.db.session import session_scope

log = logging.getLogger(__name__)
router = APIRouter()


class ChatIn(BaseModel):
    # The customer is identified by the API key; if sent, this must match it.
    customer_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class ChatOut(BaseModel):
    thread_id: str
    status: str  # completed | awaiting_approval
    reply: str | None = None
    intent: str | None = None
    ticket_id: int | None = None
    pending_approval: dict | None = None


class DecisionIn(BaseModel):
    approve: bool
    note: str | None = None


class TicketPatch(BaseModel):
    status: Literal["open", "escalated", "resolved"]


class ThreadOut(ChatOut):
    # latest approval on the thread (None if none was ever required)
    approval: dict | None = None


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_settings().graph_recursion_limit,
    }


def _pending_interrupt(graph, thread_id: str) -> dict | None:
    snap = graph.get_state(_config(thread_id))
    for task in snap.tasks:
        for itr in task.interrupts:
            return itr.value
    return None


def _view(graph, thread_id: str) -> ChatOut:
    """Read-only response built from the checkpointed state of the thread (no DB writes)."""
    pending = _pending_interrupt(graph, thread_id)
    values = graph.get_state(_config(thread_id)).values
    if pending:
        return ChatOut(
            thread_id=thread_id,
            status="awaiting_approval",
            reply="Your request needs a manager's approval. We'll update you shortly.",
            intent=values.get("intent"),
            pending_approval=pending,
        )
    return ChatOut(
        thread_id=thread_id,
        status="completed",
        reply=values.get("reply"),
        intent=values.get("intent"),
        ticket_id=values.get("ticket_id"),
    )


def _result(graph, thread_id: str) -> ChatOut:
    """Like _view, but also records the pending approval row for a paused thread."""
    out = _view(graph, thread_id)
    if out.status == "awaiting_approval":
        values = graph.get_state(_config(thread_id)).values
        with session_scope() as s:
            exists = (
                s.query(Approval)
                .filter(Approval.thread_id == thread_id, Approval.status == "pending")
                .first()
            )
            if not exists:  # idempotent: one pending row per paused thread
                s.add(
                    Approval(
                        thread_id=thread_id,
                        customer_id=values.get("customer_id", ""),
                        action=(out.pending_approval or {}).get("action", {}),
                    )
                )
    return out


@router.post("/chat", response_model=ChatOut, tags=["chat"])
def chat(body: ChatIn, request: Request, me: Principal = Depends(require_customer)):
    cid = me.customer_id
    if body.customer_id is not None and body.customer_id != cid:
        raise HTTPException(403, "customer_id does not match your API key")
    graph = request.app.state.graph
    with session_scope() as s:
        if not s.get(Customer, cid):  # key outlived its customer
            raise HTTPException(403, "customer account not found")

    thread_id = body.thread_id or str(uuid.uuid4())
    snap = graph.get_state(_config(thread_id))
    if snap.values:
        if snap.values.get("customer_id") != cid:
            raise HTTPException(403, "thread belongs to a different customer")
        if _pending_interrupt(graph, thread_id):
            raise HTTPException(409, "this thread is awaiting a manager's approval")

    with trace.turn(thread_id, "chat", customer_id=cid, actor=me.name) as info:
        graph.invoke(
            {"messages": [HumanMessage(content=body.message)], "customer_id": cid},
            _config(thread_id),
        )
        out = _result(graph, thread_id)
        info.update(status=out.status, intent=out.intent)
    return out


@router.get("/approvals", tags=["approvals"])
def list_approvals(status: str = "pending", limit: int = Query(50, ge=1, le=200),
                   offset: int = Query(0, ge=0), me: Principal = Depends(require_reviewer)):
    with session_scope() as s:
        rows = (s.query(Approval).filter(Approval.status == status).order_by(Approval.id)
                .limit(limit).offset(offset).all())
        return [
            {
                "thread_id": r.thread_id,
                "customer_id": r.customer_id,
                "action": r.action,
                "status": r.status,
                "note": r.note,
                "created_at": r.created_at,
                "decided_at": r.decided_at,
                "decided_by": r.decided_by,
            }
            for r in rows
        ]


@router.post("/approvals/{thread_id}", response_model=ChatOut, tags=["approvals"])
def decide(thread_id: str, body: DecisionIn, request: Request,
           me: Principal = Depends(require_reviewer)):
    graph = request.app.state.graph
    with session_scope() as s:
        row = (
            s.query(Approval)
            .filter(Approval.thread_id == thread_id, Approval.status == "pending")
            .first()
        )
        if not row:
            raise HTTPException(404, "no pending approval for this thread")
    if not _pending_interrupt(graph, thread_id):
        raise HTTPException(409, "thread is not paused")

    with trace.turn(thread_id, "approval", customer_id=row.customer_id, actor=me.name) as info:
        graph.invoke(Command(resume={"approved": body.approve, "note": body.note}),
                     _config(thread_id))
        info.update(status="completed",
                    intent=graph.get_state(_config(thread_id)).values.get("intent"))

    with session_scope() as s:
        row = s.get(Approval, row.id)
        row.status = "approved" if body.approve else "rejected"
        row.note = body.note
        row.decided_at = datetime.now(timezone.utc)
        row.decided_by = me.name
    log.info("approval %s on thread %s by %s", row.status, thread_id, me.name)
    return _result(graph, thread_id)


def _ticket_dict(t: Ticket) -> dict:
    return {
        "id": t.id,
        "customer_id": t.customer_id,
        "order_id": t.order_id,
        "category": t.category,
        "description": t.description,
        "priority": t.priority,
        "status": t.status,
        "created_at": t.created_at,
    }


@router.get("/tickets", tags=["tickets"])
def list_tickets(customer_id: str | None = None, status: str | None = None,
                 limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                 me: Principal = Depends(get_principal)):
    """Reviewers see all tickets (optionally filtered); customers only their own."""
    if me.role == "customer":
        if customer_id and customer_id != me.customer_id:
            raise HTTPException(403, "customers can only list their own tickets")
        customer_id = me.customer_id
    with session_scope() as s:
        q = s.query(Ticket)
        if customer_id:
            q = q.filter(Ticket.customer_id == customer_id)
        if status:
            q = q.filter(Ticket.status == status)
        return [_ticket_dict(t) for t in q.order_by(Ticket.id).limit(limit).offset(offset).all()]


@router.get("/tickets/{ticket_id}", tags=["tickets"])
def get_ticket(ticket_id: int, me: Principal = Depends(get_principal)):
    with session_scope() as s:
        t = s.get(Ticket, ticket_id)
        # 404 (not 403) for other customers' tickets: don't reveal that they exist
        if not t or (me.role == "customer" and t.customer_id != me.customer_id):
            raise HTTPException(404, "ticket not found")
        return _ticket_dict(t)


@router.patch("/tickets/{ticket_id}", tags=["tickets"])
def update_ticket(ticket_id: int, body: TicketPatch, me: Principal = Depends(require_reviewer)):
    with session_scope() as s:
        t = s.get(Ticket, ticket_id)
        if not t:
            raise HTTPException(404, "ticket not found")
        old, t.status = t.status, body.status
        out = _ticket_dict(t)
    log.info("ticket %s: %s -> %s by %s", ticket_id, old, body.status, me.name)
    return out


@router.get("/threads/{thread_id}", response_model=ThreadOut, tags=["chat"])
def thread_status(thread_id: str, request: Request, me: Principal = Depends(get_principal)):
    """Current state of a thread, e.g. whether a refund is still awaiting approval and how
    the decision went. Customers: their own threads only (404 for anyone else's)."""
    graph = request.app.state.graph
    values = graph.get_state(_config(thread_id)).values
    if not values or (me.role == "customer" and values.get("customer_id") != me.customer_id):
        raise HTTPException(404, "thread not found")
    out = _view(graph, thread_id)
    with session_scope() as s:
        row = (s.query(Approval).filter(Approval.thread_id == thread_id)
               .order_by(Approval.id.desc()).first())
        approval = None
        if row:
            approval = {"status": row.status, "created_at": row.created_at,
                        "decided_at": row.decided_at}
            if me.role == "reviewer":  # reviewer-only detail
                approval.update(note=row.note, decided_by=row.decided_by)
    return ThreadOut(**out.model_dump(), approval=approval)


@router.get("/threads/{thread_id}/trace", tags=["audit"])
def thread_trace(thread_id: str, me: Principal = Depends(get_principal)):
    """Per-turn audit trail of a thread. Reviewers: any thread; customers: their own."""
    with session_scope() as s:
        rows = s.query(TurnTrace).filter(TurnTrace.thread_id == thread_id).order_by(TurnTrace.id).all()
        # 404 (not 403) for other customers' threads: don't reveal that they exist
        if not rows or (me.role == "customer" and rows[0].customer_id != me.customer_id):
            raise HTTPException(404, "no trace for this thread")
        return {
            "thread_id": thread_id,
            "turns": [
                {
                    "id": r.id, "kind": r.kind, "actor": r.actor, "intent": r.intent,
                    "status": r.status, "started_at": r.started_at, "latency_ms": r.latency_ms,
                    "llm_calls": r.llm_calls, "node_path": r.node_path,
                    "tool_calls": r.tool_calls, "errors": r.errors,
                }
                for r in rows
            ],
        }


@router.get("/health", tags=["meta"])
def health(request: Request):
    return {"status": "ok", "rag_backend": getattr(request.app.state, "rag_backend", None)}
