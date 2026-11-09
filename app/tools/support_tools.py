"""LangChain tools. Every tool returns a JSON-serialisable dict and NEVER raises:
failures come back as {"error": ...} so the agent can react (or escalate).

Read-only tools are bound to one customer via closures, so the LLM cannot
read another customer's data by passing a different id.
"""
import logging

from langchain_core.tools import tool

from app.db.models import Customer, Order, Refund, Ticket
from app.db.session import session_scope
from app.integrations.external_client import ExternalAPIError, get_client

log = logging.getLogger(__name__)


def _order_dict(o: Order) -> dict:
    return {"order_id": o.id, "items": o.items, "total": o.total, "status": o.status,
            "placed_at": o.placed_at.date().isoformat()}


# ---- plain functions (also used directly by deterministic graph nodes) ----

def lookup_customer(customer_id: str) -> dict:
    with session_scope() as s:
        c = s.get(Customer, customer_id)
        if not c:
            return {"error": f"customer {customer_id} not found"}
        return {"customer_id": c.id, "name": c.name, "email": c.email, "tier": c.tier}


def lookup_orders(customer_id: str) -> dict:
    with session_scope() as s:
        orders = s.query(Order).filter(Order.customer_id == customer_id).all()
        return {"orders": [_order_dict(o) for o in orders]}


def lookup_order_status(customer_id: str, order_id: str) -> dict:
    with session_scope() as s:
        o = s.get(Order, order_id)
        if not o or o.customer_id != customer_id:
            return {"error": f"order {order_id} not found for this customer"}
        result = _order_dict(o)
    try:
        result["shipping"] = get_client().get_shipping(order_id)
    except ExternalAPIError as e:  # degrade gracefully: DB data is still useful
        log.warning("shipping lookup failed for %s: %s", order_id, e)
        result["shipping"] = {"error": "live shipping info temporarily unavailable"}
    return result


def create_ticket(customer_id: str, category: str, description: str,
                  order_id: str | None = None, priority: str = "medium",
                  status: str = "open") -> dict:
    with session_scope() as s:
        t = Ticket(customer_id=customer_id, order_id=order_id, category=category,
                   description=description, priority=priority, status=status)
        s.add(t)
        s.flush()
        return {"ticket_id": t.id, "status": t.status, "priority": t.priority}


def process_refund(customer_id: str, order_id: str, amount: float, reason: str) -> dict:
    """Validates, calls the external payments API, records the refund."""
    if amount <= 0:
        return {"error": "refund amount must be positive"}
    with session_scope() as s:
        o = s.get(Order, order_id)
        if not o or o.customer_id != customer_id:
            return {"error": f"order {order_id} not found for this customer"}
        already = sum(r.amount for r in s.query(Refund).filter(Refund.order_id == order_id))
        if amount + already > o.total + 1e-9:
            return {"error": f"refund exceeds order total ({o.total}); already refunded {already}"}
    try:
        ext = get_client().create_refund(order_id, amount, reason)
    except ExternalAPIError as e:
        return {"error": f"payment provider unavailable: {e}"}
    with session_scope() as s:
        s.add(Refund(order_id=order_id, amount=amount, reason=reason,
                     external_ref=ext.get("refund_ref")))
    return {"refund_ref": ext.get("refund_ref"), "amount": amount, "order_id": order_id}


# ---- LLM-facing tools ----

def make_account_tools(customer_id: str) -> list:
    @tool
    def get_customer_profile() -> dict:
        """Get the current customer's profile (name, email, tier)."""
        return lookup_customer(customer_id)

    @tool
    def list_my_orders() -> dict:
        """List all orders of the current customer with status and totals."""
        return lookup_orders(customer_id)

    @tool
    def get_order_status(order_id: str) -> dict:
        """Get details and live shipping/tracking status for one order (e.g. 'ORD-1001')."""
        return lookup_order_status(customer_id, order_id)

    return [get_customer_profile, list_my_orders, get_order_status]
