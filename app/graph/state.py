from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class SupportState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    customer_id: str
    intent: str  # policy_question | account_query | complaint | refund_request | other
    priority: str  # low | medium | high
    order_id: Optional[str]
    retrieved_docs: list[dict]
    account_summary: str
    proposed_action: Optional[dict]  # {action: refund|complaint|escalate|none, ...}
    approval_status: Optional[str]  # not_required | approved | rejected
    approval_note: Optional[str]
    ticket_id: Optional[int]
    reply: str
    errors: list[str]
