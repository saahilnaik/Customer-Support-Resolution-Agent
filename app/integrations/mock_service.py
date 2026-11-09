"""Mock 'third-party' logistics + payments API, mounted at /mock.

POST /mock/_chaos {"fail": true} makes every call return 503, so you can demo
retries and graceful degradation.
"""
import hashlib

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/mock", tags=["mock-external-api"])

_state = {"fail": False}
_refunds: dict[str, dict] = {}  # idempotency key -> refund

_SHIPPING = {
    "ORD-1001": {"carrier": "UPS", "status": "delivered", "eta": None},
    "ORD-1002": {"carrier": "FedEx", "status": "in_transit", "eta": "2 days"},
    "ORD-2001": {"carrier": "UPS", "status": "delivered", "eta": None},
    "ORD-2002": {"carrier": "USPS", "status": "delivered", "eta": None},
    "ORD-3001": {"carrier": None, "status": "label_pending", "eta": "5 days"},
}


def _check_chaos():
    if _state["fail"]:
        raise HTTPException(503, "mock service unavailable (chaos mode)")


class Chaos(BaseModel):
    fail: bool


@router.post("/_chaos")
def set_chaos(body: Chaos):
    _state["fail"] = body.fail
    return _state


@router.get("/orders/{order_id}/shipping")
def shipping(order_id: str):
    _check_chaos()
    info = _SHIPPING.get(order_id)
    if not info:
        raise HTTPException(404, "unknown order")
    tracking = "1Z" + hashlib.md5(order_id.encode()).hexdigest()[:10].upper()
    return {"order_id": order_id, "tracking_number": tracking, **info}


class RefundIn(BaseModel):
    order_id: str
    amount: float
    reason: str


@router.post("/refunds")
def refund(body: RefundIn, idempotency_key: str | None = Header(default=None)):
    _check_chaos()
    key = idempotency_key or f"{body.order_id}-{body.amount}"
    if key in _refunds:
        return _refunds[key]
    result = {
        "refund_ref": "RF-" + hashlib.md5(key.encode()).hexdigest()[:8].upper(),
        "order_id": body.order_id,
        "amount": body.amount,
        "status": "processed",
    }
    _refunds[key] = result
    return result
