import functools
import json
import logging
import re

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

from app import llm as llm_mod

log = logging.getLogger(__name__)

HISTORY_WINDOW = 10


def history(state, n: int = HISTORY_WINDOW) -> list:
    """Recent conversation, excluding tool noise."""
    return [m for m in state.get("messages", []) if m.type in ("human", "ai")][-n:]


def last_user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if m.type == "human":
            return str(m.content)
    return ""


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (tolerates code fences / prose)."""
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in LLM output: {str(text)[:200]!r}")
    return json.loads(m.group(0))


def reply_update(text: str, **extra) -> dict:
    return {"reply": text, "messages": [AIMessage(content=text)], **extra}


def safe_node(fn):
    """Node-level fallback: any unexpected failure becomes an escalation
    proposal instead of a 500. Graph interrupts must pass through untouched."""

    @functools.wraps(fn)
    def wrapper(state):
        try:
            return fn(state)
        except GraphBubbleUp:
            raise
        except Exception as e:
            log.exception("node %s failed", fn.__name__)
            return {
                "errors": [f"{fn.__name__}: {type(e).__name__}: {e}"],
                "proposed_action": {
                    "action": "escalate",
                    "reason": f"Automated handling failed in '{fn.__name__}': {e}",
                },
            }

    return wrapper


def run_tool_loop(system: str, msgs: list, tools: list, max_steps: int = 5) -> str:
    """Minimal tool-calling agent loop. Tool failures are returned to the model."""
    llm = llm_mod.get_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    convo = [SystemMessage(content=system), *msgs]
    for _ in range(max_steps):
        ai = llm_mod.invoke_llm(llm, convo)
        convo.append(ai)
        if not ai.tool_calls:
            return str(ai.content)
        for call in ai.tool_calls:
            tool = tool_map.get(call["name"])
            if tool is None:
                result = {"error": f"unknown tool {call['name']}"}
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as e:  # tools shouldn't raise, but never trust that
                    result = {"error": f"{type(e).__name__}: {e}"}
            convo.append(
                ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
    raise RuntimeError("tool loop exceeded max steps")
