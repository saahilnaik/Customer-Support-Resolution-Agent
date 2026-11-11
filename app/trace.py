"""Per-turn audit trace: node path, LLM calls, tool calls, errors, latency.

A `Trace` is bound to a contextvar for the duration of one graph invocation.
Instrumentation points (`traced_node`, `invoke_llm`, `run_tool_loop`) record into
it if one is active and are no-ops otherwise. Persisting a trace never raises:
observability must not break request handling.
"""
import contextlib
import contextvars
import functools
import logging
import time
from datetime import datetime, timezone

from langgraph.errors import GraphBubbleUp

from app.db.models import TurnTrace
from app.db.session import session_scope

log = logging.getLogger(__name__)
_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)


class Trace:
    def __init__(self):
        self.node_path: list[dict] = []
        self.tool_calls: list[dict] = []
        self.errors: list[str] = []
        self.llm_calls = 0


def current() -> "Trace | None":
    return _current.get()


def record_llm_call() -> None:
    if (t := current()):
        t.llm_calls += 1


def record_tool_call(name: str, args: dict, ok: bool, latency_ms: float) -> None:
    if (t := current()):
        t.tool_calls.append({"name": name, "args": args, "ok": ok, "latency_ms": round(latency_ms, 1)})


def traced_node(name: str, fn):
    """Wrap a graph node so its execution (path, latency, outcome, errors) is recorded."""

    @functools.wraps(fn)
    def wrapper(state):
        t = current()
        if t is None:
            return fn(state)
        start = time.perf_counter()
        entry = {"node": name, "latency_ms": 0.0, "outcome": "ok"}
        t.node_path.append(entry)
        try:
            out = fn(state)
            if isinstance(out, dict) and out.get("errors"):
                entry["outcome"] = "error"
                t.errors.extend(str(e) for e in out["errors"])
            return out
        except GraphBubbleUp:
            entry["outcome"] = "interrupted"
            raise
        except Exception as e:
            entry["outcome"] = "error"
            t.errors.append(f"{name}: {type(e).__name__}: {e}")
            raise
        finally:
            entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)

    return wrapper


@contextlib.contextmanager
def turn(thread_id: str, kind: str, customer_id: str | None = None, actor: str | None = None):
    """Trace one graph invocation. Yields a dict; set `status`/`intent` on it before exit."""
    t = Trace()
    token = _current.set(t)
    info: dict = {}
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        yield info
    except Exception as e:
        info.setdefault("status", "failed")
        t.errors.append(f"{type(e).__name__}: {e}")
        raise
    finally:
        _current.reset(token)
        try:
            with session_scope() as s:
                s.add(TurnTrace(
                    thread_id=thread_id, customer_id=customer_id, kind=kind, actor=actor,
                    intent=info.get("intent"), status=info.get("status"), started_at=started,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    llm_calls=t.llm_calls, node_path=t.node_path,
                    tool_calls=t.tool_calls, errors=t.errors,
                ))
        except Exception:
            log.exception("failed to persist trace for thread %s", thread_id)
