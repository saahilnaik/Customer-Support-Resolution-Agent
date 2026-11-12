"""LangChain tools. Every tool returns a JSON-serialisable dict and NEVER raises:
failures come back as {"error": ...} so the agent can react (or escalate).

Read-only tools are bound to one customer via closures, so the LLM cannot
read another customer's data by passing a different id.
"""
import logging
from datetime import datetime, timedelta, timezone

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


REFUND_WINDOW_DAYS = 30


def check_refund_eligibility(customer_id: str, order_id: str) -> dict:
    """Deterministic refund-window policy (the LLM never decides this).

    Delivered orders can be refunded within REFUND_WINDOW_DAYS. We have no delivery
    timestamp, so the order date is the clock. Orders not yet delivered are not blocked,
    and unknown/foreign orders are left to process_refund's own error.
    Returns {"allowed", "reason"}; "reason" is customer-facing when not allowed.
    """
    with session_scope() as s:
        o = s.get(Order, order_id)
        if not o or o.customer_id != customer_id or o.status != "delivered":
            return {"allowed": True, "reason": ""}
        placed = o.placed_at if o.placed_at.tzinfo else o.placed_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - placed
    if age > timedelta(days=REFUND_WINDOW_DAYS):
        return {"allowed": False,
                "reason": f"Order {order_id} is outside our {REFUND_WINDOW_DAYS}-day refund "
                          "window. If the item is faulty it may still be covered by the "
                          "12-month warranty (replacement or repair)."}
    return {"allowed": True, "reason": ""}


def process_refund(customer_id: str, order_id: str, amount: float, reason: str,
                   enforce_window: bool = True) -> dict:
    """Validates, calls the external payments API, records the refund.
    cancel_order passes enforce_window=False: refunding a cancelled order isn't a return."""
    if amount <= 0:
        return {"error": "refund amount must be positive"}
    if enforce_window:
        check = check_refund_eligibility(customer_id, order_id)
        if not check["allowed"]:
            return {"error": check["reason"]}
    with session_scope() as s:
        o = s.get(Order, order_id)
        if not o or o.customer_id != customer_id:
            return {"error": f"order {order_id} not found for this customer"}
        prior = s.query(Refund).filter(Refund.order_id == order_id).all()
        already = sum(r.amount for r in prior)
        if amount + already > o.total + 1e-9:
            return {"error": f"refund exceeds order total ({o.total}); already refunded {already}"}
        # Nth refund on this order -> its own key; HTTP retries of this call reuse it.
        key = f"refund-{order_id}-{len(prior) + 1}"
    try:
        ext = get_client().create_refund(order_id, amount, reason, idempotency_key=key)
    except ExternalAPIError as e:
        return {"error": f"payment provider unavailable: {e}"}
    with session_scope() as s:
        s.add(Refund(order_id=order_id, amount=amount, reason=reason,
                     external_ref=ext.get("refund_ref")))
    return {"refund_ref": ext.get("refund_ref"), "amount": amount, "order_id": order_id}


# ---- order changes (cancel / shipping address) ----

ORDER_CHANGE_ACTIONS = {"cancel_order", "change_address"}


def check_order_change(customer_id: str, order_id: str, action: str) -> dict:
    """Deterministic eligibility policy for order changes (the LLM never decides this).

    Returns {"allowed", "requires_approval", "reason", "status", "total"}; "reason"
    is customer-facing when not allowed and reviewer-facing when approval is needed.
    """
    deny = {"allowed": False, "requires_approval": False}
    if action not in ORDER_CHANGE_ACTIONS:
        return {**deny, "reason": f"unsupported order change {action!r}"}
    with session_scope() as s:
        o = s.get(Order, order_id)
        if not o or o.customer_id != customer_id:
            return {**deny, "reason": f"I couldn't find order {order_id} on your account."}
        status, total = o.status, o.total
    base = {"status": status, "total": total}

    if status == "cancelled":
        return {**deny, **base, "reason": f"Order {order_id} is already cancelled."}
    if action == "cancel_order":
        if status == "processing":
            return {"allowed": True, "requires_approval": False, "reason": "", **base}
        if status == "shipped":
            return {"allowed": True, "requires_approval": True, **base,
                    "reason": f"Order {order_id} has already shipped; cancelling needs a "
                              f"carrier intercept and a ${total:.2f} refund."}
        return {**deny, **base,
                "reason": f"Order {order_id} has already been delivered, so it can't be "
                          "cancelled. You can return it within 30 days for a refund instead."}
    # change_address
    if status == "processing":
        return {"allowed": True, "requires_approval": False, "reason": "", **base}
    progress = "been delivered" if status == "delivered" else "shipped"
    return {**deny, **base,
            "reason": f"Order {order_id} has already {progress}, so its address can no longer "
                      "be changed. Please contact the carrier using the tracking number in "
                      "your shipping email."}


def cancel_order(customer_id: str, order_id: str, reason: str) -> dict:
    """Cancel via the logistics API, mark the order cancelled, refund what was paid.

    Eligibility is re-checked here because the order may have changed while the
    request waited for a manager's approval.
    """
    check = check_order_change(customer_id, order_id, "cancel_order")
    if not check["allowed"]:
        return {"error": check["reason"]}
    try:
        ext = get_client().cancel_order(order_id, reason)
    except ExternalAPIError as e:
        return {"error": f"logistics provider could not cancel the order: {e}"}
    with session_scope() as s:
        o = s.get(Order, order_id)
        o.status = "cancelled"
        already = sum(r.amount for r in s.query(Refund).filter(Refund.order_id == order_id))
        remaining = round(o.total - already, 2)
    result = {"cancelled": True, "order_id": order_id, "cancel_ref": ext.get("cancel_ref"),
              "refund_amount": 0.0, "refund_ref": None}
    if remaining > 0:
        refund = process_refund(customer_id, order_id, remaining, f"order cancelled: {reason}",
                                 enforce_window=False)
        if "error" in refund:
            # the order IS cancelled; the caller must escalate the missing refund
            return {**result, "refund_error": refund["error"]}
        result.update(refund_amount=remaining, refund_ref=refund["refund_ref"])
    return result


def change_shipping_address(customer_id: str, order_id: str, address: str) -> dict:
    address = (address or "").strip()
    if not address:
        return {"error": "a new shipping address is required"}
    check = check_order_change(customer_id, order_id, "change_address")
    if not check["allowed"]:
        return {"error": check["reason"]}
    try:
        get_client().update_address(order_id, address)
    except ExternalAPIError as e:
        return {"error": f"logistics provider could not update the address: {e}"}
    return {"order_id": order_id, "address": address}


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
