from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import SupportState


def _failed(state) -> bool:
    """A node's safe_node fallback set errors + an escalation proposal."""
    return bool(state.get("errors")) and bool(state.get("proposed_action"))


def route_after_supervisor(state) -> str:
    if _failed(state):
        return "execute"
    return {
        "policy_question": "knowledge",
        "account_query": "account",
        "refund_request": "account",  # gather order facts first
        "complaint": "resolution_plan",
        "order_change": "order_change",
    }.get(state.get("intent"), "general")


def route_after_account(state) -> str:
    if _failed(state):
        return "execute"
    return "resolution_plan" if state.get("intent") == "refund_request" else END


def route_after_plan(state) -> str:
    return "execute" if _failed(state) else "approval_gate"


def build_graph(checkpointer=None):
    g = StateGraph(SupportState)
    g.add_node("supervisor", nodes.supervisor)
    g.add_node("knowledge", nodes.knowledge)
    g.add_node("account", nodes.account)
    g.add_node("general", nodes.general)
    g.add_node("resolution_plan", nodes.resolution_plan)
    g.add_node("order_change", nodes.order_change)
    g.add_node("approval_gate", nodes.approval_gate)
    g.add_node("execute", nodes.execute)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        ["knowledge", "account", "resolution_plan", "order_change", "general", "execute"],
    )
    g.add_conditional_edges("account", route_after_account, ["resolution_plan", "execute", END])
    # both "proposer" nodes feed the same approval gate -> execute pipeline
    for proposer in ("resolution_plan", "order_change"):
        g.add_conditional_edges(proposer, route_after_plan, ["approval_gate", "execute"])
    g.add_edge("approval_gate", "execute")
    for n in ("knowledge", "general", "execute"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer)
