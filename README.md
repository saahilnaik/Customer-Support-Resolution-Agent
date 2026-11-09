# Customer Support Resolution Agent

A multi-agent customer-support system built with **LangGraph + FastAPI**, free to run
(Groq free tier, local embeddings, SQLite).

| Capability | Where |
|---|---|
| Multi-agent workflow | `app/graph/builder.py`: supervisor routes to knowledge / account / resolution agents |
| Tool calling | `app/tools/support_tools.py`, loop in `app/graph/common.py:run_tool_loop` (customer-scoped tools) |
| API integration | `app/integrations/external_client.py` (httpx + retries + idempotency key) against a mock service at `/mock` |
| RAG | `app/rag/retriever.py`: local fastembed embeddings (TF-IDF fallback) over `data/kb/*.md`, cited answers, "I don't know" on low confidence |
| Human-in-the-loop | refunds over `$50` call `interrupt()` in `approval_gate`; state is checkpointed in SQLite; resume via `POST /approvals/{thread_id}` |
| Error handling | LLM retry/backoff, external API retry, `safe_node` fallback to escalation, graceful degradation, structured errors from tools, global 500 handler |

## Graph

```
START -> supervisor --policy_question--> knowledge -> END
                    --account_query----> account ---------------> END
                    --refund_request--> account -> resolution_plan -> approval_gate -> execute -> END
                    --complaint--------> resolution_plan ------------^ (interrupt if refund > threshold)
                    --other------------> general -> END
(any node failure -> escalation proposal -> execute -> ticket for a human)
```

The LLM only *proposes* an action (`resolution_plan`); `execute` performs it deterministically after
the approval gate, so side effects never run on an unapproved or replayed step.

## Run

1. **Prerequisites:** Python 3.11+ and a free Groq API key (https://console.groq.com).
2. **Create a virtualenv and install dependencies** (from the project folder):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure:** copy `.env.example` to `.env` (`copy .env.example .env`, or `cp` on macOS/Linux) and set
   `GROQ_API_KEY`. `.env` is git-ignored.
4. **Start the server:**
   ```powershell
   python -m uvicorn app.main:app --reload
   ```
   Wait for `Application startup complete`. The first start downloads the small embedding model
   (~130 MB) once; offline it falls back to TF-IDF. The SQLite databases are created in `data/`.
5. **Try it:** open http://127.0.0.1:8000/docs and use the Swagger UI, or run the curl flows below.
   Seeded customers: `C001` (Alice), `C002` (Bob), `C003` (Carla); orders `ORD-1001`, `ORD-1002`,
   `ORD-2001`, `ORD-2002`, `ORD-3001`.
6. **Run the tests** (no API key or network needed): `python -m pytest -q`.
7. **Reset demo data:** stop the server and delete `data/support.db` and `data/checkpoints.db*`.

### Troubleshooting
- **Every chat is answered with "passed this to a human agent":** the LLM call is failing and the agent
  is escalating by design. Check the server log for the real error (missing key, `model_not_found`, rate limit).
- **`model_not_found`:** your key can't use the configured model. List the models you can use and set
  `GROQ_MODEL` in `.env` to one with good tool calling:
  ```powershell
  python -c "import httpx;from app.config import get_settings as g;print([m['id'] for m in httpx.get('https://api.groq.com/openai/v1/models',headers={'Authorization':'Bearer '+g().groq_api_key}).json()['data']])"
  ```
- **Changed `.env`?** `--reload` only watches `.py` files; restart the server. Also wait for startup to
  finish after any reload before sending requests.

## Demo flows

```bash
# 1. Policy question (RAG)
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_id":"C001","message":"How long do I have to return an item?"}'

# 2. Order status (tool calling + external shipping API)
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_id":"C001","message":"Where is order ORD-1002?"}'

# 3. Complaint -> ticket
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_id":"C001","message":"My headphones from ORD-1001 stopped working"}'

# 4. Large refund -> awaiting_approval, then a manager decides
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_id":"C001","message":"I want a full refund for ORD-1001, the headphones are defective"}'
curl localhost:8000/approvals
curl -X POST localhost:8000/approvals/<thread_id> -H 'content-type: application/json' \
  -d '{"approve":true,"note":"verified defect"}'

# 5. Failure handling: make the external API return 503, then retry a refund / order lookup
curl -X POST localhost:8000/mock/_chaos -H 'content-type: application/json' -d '{"fail":true}'
```

Reuse the returned `thread_id` in `/chat` to continue a conversation.

## Tests

```bash
pytest
```
The LLM is replaced by a scripted fake model and the external API by a fake transport, so tests need no
network or API key. They cover tool scoping, refund validation, client retries, RAG, the full graph
(auto-approved refund, approve, reject, planner garbage output, provider outage, LLM rate-limit exhaustion).

## Limitations
This is a reference/portfolio build, not a production system.
- **No authentication or authorization.** Anyone who can reach the API can chat as any customer and
  approve or reject refunds. A real deployment needs customer auth and reviewer login with roles.
- **Mock backends.** The "external" order/shipping/payment API is a local mock, customers and orders are
  seeded demo data in SQLite, and the knowledge base is four short markdown files.
- **LLM behaviour is not systematically evaluated.** Tests use a scripted fake model, so they verify the
  graph, approval flow and error handling, not how well a real model classifies or plans. Manual runs showed
  the planner can misjudge a case (for example escalating a large refund itself), which prompt rules reduce but
  do not eliminate. Only refunds are approval-gated; the model never moves money directly.
- **Single process, local storage.** SQLite and an in-memory vector index suit a demo, not concurrent
  multi-instance deployment. There is no queue, rate limiting of API callers, or observability stack.
- **Free-tier constraints.** Groq's free tier is rate limited and its available models change; calls are
  retried with backoff, but a long demo can still be throttled and will then escalate.
- **Notifications.** A rejected or pending approval is returned in the API response only; the customer is
  not emailed.

## Notes
- Windows with Application Control may block SQLAlchemy's compiled extensions; deleting the
  `*.pyd` files under `site-packages/sqlalchemy` makes it use the pure-Python fallback.
- Model and thresholds are config (`.env`): `GROQ_MODEL`, `REFUND_APPROVAL_THRESHOLD`, `EMBEDDING_BACKEND`.
- Free-tier rate limits are the main constraint; a typical turn uses 2-4 LLM calls.
