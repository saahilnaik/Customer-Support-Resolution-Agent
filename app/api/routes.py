import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.models import Approval, Customer, Ticket
from app.db.session import session_scope

log = logging.getLogger(__name__)
router = APIRouter()


class ChatIn(BaseModel):
    customer_id: str
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


def _result(graph, thread_id: str) -> ChatOut:
    """Build the API response from the checkpointed state of the thread."""
    pending = _pending_interrupt(graph, thread_id)
    values = graph.get_state(_config(thread_id)).values
    if pending:
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
                        action=pending.get("action", {}),
                    )
                )
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


@router.post("/chat", response_model=ChatOut, tags=["chat"])
def chat(body: ChatIn, request: Request):
    graph = request.app.state.graph
    with session_scope() as s:
        if not s.get(Customer, body.customer_id):
            raise HTTPException(404, f"customer {body.customer_id} not found")

    thread_id = body.thread_id or str(uuid.uuid4())
    snap = graph.get_state(_config(thread_id))
    if snap.values:
        if snap.values.get("customer_id") != body.customer_id:
            raise HTTPException(403, "thread belongs to a different customer")
        if _pending_interrupt(graph, thread_id):
            raise HTTPException(409, "this thread is awaiting a manager's approval")

    graph.invoke(
        {"messages": [HumanMessage(content=body.message)], "customer_id": body.customer_id},
        _config(thread_id),
    )
    return _result(graph, thread_id)


@router.get("/approvals", tags=["approvals"])
def list_approvals(status: str = "pending"):
    with session_scope() as s:
        rows = s.query(Approval).filter(Approval.status == status).order_by(Approval.id).all()
        return [
            {
                "thread_id": r.thread_id,
                "customer_id": r.customer_id,
                "action": r.action,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in rows
        ]


@router.post("/approvals/{thread_id}", response_model=ChatOut, tags=["approvals"])
def decide(thread_id: str, body: DecisionIn, request: Request):
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

    graph.invoke(Command(resume={"approved": body.approve, "note": body.note}), _config(thread_id))

    with session_scope() as s:
        row = s.get(Approval, row.id)
        row.status = "approved" if body.approve else "rejected"
        row.note = body.note
        row.decided_at = datetime.now(timezone.utc)
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
def list_tickets(customer_id: str | None = None):
    with session_scope() as s:
        q = s.query(Ticket)
        if customer_id:
            q = q.filter(Ticket.customer_id == customer_id)
        return [_ticket_dict(t) for t in q.order_by(Ticket.id).all()]


@router.get("/tickets/{ticket_id}", tags=["tickets"])
def get_ticket(ticket_id: int):
    with session_scope() as s:
        t = s.get(Ticket, ticket_id)
        if not t:
            raise HTTPException(404, "ticket not found")
        return _ticket_dict(t)


@router.get("/health", tags=["meta"])
def health(request: Request):
    return {"status": "ok", "rag_backend": getattr(request.app.state, "rag_backend", None)}
