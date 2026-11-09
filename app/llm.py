"""LLM factory (Groq) + retrying invoke helper."""
import logging

from langchain_core.language_models import BaseChatModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

log = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    from langchain_groq import ChatGroq

    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (get a free key at console.groq.com)")
    # max_retries=0: retrying is handled by invoke_llm so it is logged and bounded
    return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0, max_retries=0)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    return (
        name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
        or status in {429, 500, 502, 503, 504}
    )


def invoke_llm(llm, messages):
    """llm.invoke with exponential backoff on rate limits / transient errors."""
    s = get_settings()

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(s.llm_retry_attempts),
        wait=wait_exponential(multiplier=1, max=s.llm_retry_max_wait),
        before_sleep=lambda rs: log.warning(
            "LLM call failed (attempt %s), retrying: %s", rs.attempt_number, rs.outcome.exception()
        ),
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)

    return _call()
