import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="support-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["CHECKPOINT_DB"] = f"{_tmp}/ckpt.db"
os.environ["EMBEDDING_BACKEND"] = "tfidf"  # no model download in tests
os.environ["GROQ_API_KEY"] = "test"
os.environ["EXTERNAL_BACKOFF"] = "0"

import httpx  # noqa: E402
import pytest  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from app import llm as llm_mod  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.seed import seed  # noqa: E402
from app.db.session import get_engine, init_db  # noqa: E402
from app.integrations.external_client import ExternalClient, set_client  # noqa: E402


class ScriptedChat(BaseChatModel):
    """Returns pre-scripted AIMessages in order. Records how many were consumed."""

    script: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.calls >= len(self.script):
            raise AssertionError("LLM called more times than scripted")
        msg = self.script[self.calls]
        self.calls += 1
        if isinstance(msg, Exception):
            raise msg
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture(autouse=True)
def fresh_db():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    init_db()
    seed()


@pytest.fixture
def script_llm(monkeypatch):
    def _install(*messages) -> ScriptedChat:
        fake = ScriptedChat(script=list(messages))
        monkeypatch.setattr(llm_mod, "get_llm", lambda: fake)
        return fake

    return _install


@pytest.fixture
def external(monkeypatch):
    """Install an ExternalClient backed by a fake transport. Returns a state dict."""
    state = {"fail": False, "refunds": [], "shipping_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503, json={"detail": "down"})
        if request.url.path.endswith("/shipping"):
            state["shipping_calls"] += 1
            return httpx.Response(200, json={"status": "in_transit", "carrier": "FedEx"})
        if request.url.path.endswith("/refunds"):
            body = __import__("json").loads(request.content)
            state["refunds"].append(body)
            return httpx.Response(200, json={"refund_ref": "RF-TEST", **body})
        return httpx.Response(404)

    http = httpx.Client(base_url="http://ext/", transport=httpx.MockTransport(handler))
    set_client(ExternalClient(http=http, attempts=3, backoff=0))
    yield state
    set_client(None)
