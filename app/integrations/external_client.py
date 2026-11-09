"""HTTP client for the external order/shipping/refund API, with retries.

5xx, 429 and transport errors are retried with exponential backoff; 4xx
errors fail immediately. After retries are exhausted an ExternalAPIError is
raised, which tools convert into structured error results.
"""
import logging

import httpx
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

log = logging.getLogger(__name__)


class ExternalAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class _Retryable(Exception):
    pass


class ExternalClient:
    def __init__(self, http: httpx.Client | None = None, attempts: int | None = None,
                 backoff: float | None = None):
        s = get_settings()
        self.http = http or httpx.Client(
            base_url=s.external_api_base_url.rstrip("/") + "/", timeout=5.0
        )
        self.attempts = attempts or s.external_retry_attempts
        self.backoff = s.external_backoff if backoff is None else backoff

    def _request(self, method: str, path: str, **kw) -> dict:
        def once() -> dict:
            try:
                resp = self.http.request(method, path, **kw)
            except httpx.TransportError as e:
                raise _Retryable(f"transport error: {e}") from e
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retryable(f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise ExternalAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code
                )
            return resp.json()

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.attempts),
                wait=wait_exponential(multiplier=self.backoff, max=5),
                retry=retry_if_exception_type(_Retryable),
                before_sleep=lambda rs: log.warning(
                    "external API retry %s %s (attempt %s): %s",
                    method, path, rs.attempt_number, rs.outcome.exception(),
                ),
            ):
                with attempt:
                    return once()
        except RetryError as e:
            raise ExternalAPIError(f"external API unavailable: {e.last_attempt.exception()}") from e
        raise AssertionError("unreachable")

    def get_shipping(self, order_id: str) -> dict:
        return self._request("GET", f"orders/{order_id}/shipping")

    def create_refund(self, order_id: str, amount: float, reason: str) -> dict:
        return self._request(
            "POST", "refunds",
            json={"order_id": order_id, "amount": amount, "reason": reason},
            headers={"Idempotency-Key": f"refund-{order_id}"},  # safe to retry
        )


_client: ExternalClient | None = None


def get_client() -> ExternalClient:
    global _client
    if _client is None:
        _client = ExternalClient()
    return _client


def set_client(client: ExternalClient | None) -> None:
    global _client
    _client = client
