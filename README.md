# Customer Support Resolution Agent

A multi-agent customer-support system built with **LangGraph + FastAPI**, free to run
(Groq free tier, local embeddings, SQLite).

| Capability | Where |
|---|---|
| Multi-agent workflow | `app/graph/builder.py`: supervisor routes to knowledge / account / resolution / order-change agents |
| Order changes | `order_change` agent cancels orders or changes the shipping address; eligibility is a deterministic policy check, and cancelling a shipped order needs a manager |
| Tool calling | `app/tools/support_tools.py`, loop in `app/graph/common.py:run_tool_loop` (customer-scoped tools) |
| API integration | `app/integrations/external_client.py` (httpx + retries + idempotency key) against a mock service at `/mock` |
| RAG | `app/rag/retriever.py`: local fastembed embeddings (TF-IDF fallback) over `data/kb/*.md`, cited answers, "I don't know" on low confidence |
| Human-in-the-loop | refunds over `$50` and cancellations of shipped orders call `interrupt()` in `approval_gate`; state is checkpointed in SQLite; resume via `POST /approvals/{thread_id}` |
| Auth | `app/auth.py`: hashed API keys (`X-API-Key`), `customer` role bound to one account, `reviewer` role for approvals; decisions record who made them |
| Error handling | LLM retry/backoff, external API retry, `safe_node` fallback to escalation, graceful degradation, structured errors from tools, global 500 handler |

## Graph

```
START -> supervisor --policy_question--> knowledge -> END
                    --account_query----> account ---------------> END
                    --refund_request--> account -> resolution_plan -> approval_gate -> execute -> END
                    --complaint--------> resolution_plan ------------^ (interrupt if refund > threshold)
                    --order_change-----> order_change ---------------^ (interrupt if order already shipped)
                    --other------------> general -> END
(any node failure -> escalation proposal -> execute -> ticket for a human)
```

The LLM only *proposes* an action (`resolution_plan`, `order_change`); `execute` performs it
deterministically after the approval gate, so side effects never run on an unapproved or replayed step.
For order changes the model only extracts what the customer wants; whether it is allowed is decided in code:

| Order status | Cancel | Change address |
|---|---|---|
| processing | immediately, full refund | yes |
| shipped | manager approval (carrier intercept), then full refund | no, contact carrier |
| delivered | no, return instead | no |

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
5. **Try it:** open http://127.0.0.1:8000/docs, click **Authorize** and enter an API key (see
   [Authentication](#authentication)), or run the curl flows below.
   Seeded customers: `C001` (Alice), `C002` (Bob), `C003` (Carla); orders `ORD-1001`, `ORD-1002`,
   `ORD-2001`, `ORD-2002`, `ORD-3001`.
6. **Run the tests** (no API key or network needed): `python -m pytest -q`.
7. **Reset demo data:** stop the server and delete `data/support.db` and `data/checkpoints.db*`.

### Quick start (exact steps, Windows PowerShell)

```powershell
cd "<path-to>\Customer Support Resolution Agent"
.venv\Scripts\activate
python -m uvicorn app.main:app
```

Expected output:

```
INFO:     Waiting for application startup.
... INFO support: ready (rag=_FastEmbedder)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Then open http://127.0.0.1:8000 (redirects to the Swagger UI at `/docs`). Stop with `Ctrl+C`.
`rag=_FastEmbedder` means the local embedding model loaded; `rag=_TfidfEmbedder`-style output means the offline fallback is in use.

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

## Authentication

Every endpoint except `/health`, `/docs` and the `/mock` service needs an `X-API-Key` header.
The customer is taken from the key, so `customer_id` in the `/chat` body is optional (if sent, it must match).

| Role | Can | Cannot |
|---|---|---|
| `customer` (bound to one customer) | `POST /chat` as that customer, own threads only; `GET /tickets` (own only) | see other customers' threads/tickets, approvals |
| `reviewer` | `GET /approvals`, `POST /approvals/{thread_id}` (recorded as `decided_by`), all `/tickets` | chat |

Missing/invalid/revoked key -> `401`; wrong role or someone else's thread -> `403`; another customer's ticket -> `404`.

**Demo keys** (seeded while `SEED_DEMO_KEYS=true`, the default; set it to `false` anywhere real, these are public):
`demo-key-alice` (C001), `demo-key-bob` (C002), `demo-key-carla` (C003), `demo-key-reviewer`.

**Real keys:** keys are random and stored only as SHA-256 hashes, so they're shown once:
```powershell
python -m app.auth create-key --role reviewer --name "Dana"
python -m app.auth create-key --role customer --name "alice-mobile" --customer C001
python -m app.auth revoke --name "Dana"
```

## Demo flows

```bash
ALICE='X-API-Key: demo-key-alice'; CARLA='X-API-Key: demo-key-carla'; REVIEWER='X-API-Key: demo-key-reviewer'
JSON='content-type: application/json'

# 1. Policy question (RAG)
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" \
  -d '{"message":"How long do I have to return an item?"}'

# 2. Order status (tool calling + external shipping API)
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" -d '{"message":"Where is order ORD-1002?"}'

# 3. Complaint -> ticket
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" \
  -d '{"message":"My headphones from ORD-1001 stopped working"}'

# 4. Large refund -> awaiting_approval, then a reviewer decides
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" \
  -d '{"message":"I want a full refund for ORD-1001, the headphones are defective"}'
curl localhost:8000/approvals -H "$REVIEWER"
curl -X POST localhost:8000/approvals/<thread_id> -H "$REVIEWER" -H "$JSON" \
  -d '{"approve":true,"note":"verified defect"}'

# 5. Order changes: cancel an unshipped order (instant + refund), change an address,
#    cancel a shipped order (goes to /approvals like a large refund)
curl -X POST localhost:8000/chat -H "$CARLA" -H "$JSON" \
  -d '{"message":"Please ship ORD-3001 to 42 Elm St, Austin TX 78701"}'
curl -X POST localhost:8000/chat -H "$CARLA" -H "$JSON" -d '{"message":"Actually, cancel ORD-3001"}'
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" -d '{"message":"Cancel ORD-1002 please"}'

# 6. Auth: a customer can't approve, and can't impersonate another customer
curl localhost:8000/approvals -H "$ALICE"                                              # 403
curl -X POST localhost:8000/chat -H "$ALICE" -H "$JSON" -d '{"customer_id":"C002","message":"hi"}'  # 403

# 7. Failure handling: make the external API return 503, then retry a refund / order lookup
curl -X POST localhost:8000/mock/_chaos -H "$JSON" -d '{"fail":true}'
```

Reuse the returned `thread_id` in `/chat` to continue a conversation.

## Tests

```bash
pytest
```
The LLM is replaced by a scripted fake model and the external API by a fake transport, so tests need no
network or API key. They cover tool scoping, refund validation, client retries, RAG, the full graph
(auto-approved refund, approve, reject, planner garbage output, provider outage, LLM rate-limit exhaustion),
order changes (eligibility rules, instant cancel + refund, approval-gated cancel, rejection,
cancel succeeded but refund failed, address change, missing order number), and auth (401/403 per role,
impersonation, ticket isolation, `decided_by`, key hashing/revocation, old-schema column upgrade).

## Limitations
This is a reference/portfolio build, not a production system.
- **Simple API-key auth.** Keys identify a customer or a reviewer, but there is no user login, key
  expiry/rotation, OAuth/SSO, or per-key rate limiting, and the demo keys are public (disable with
  `SEED_DEMO_KEYS=false`). Run it behind HTTPS: keys are bearer secrets. The `/mock` service is unauthenticated.
- **Mock backends.** The "external" order/shipping/payment API is a local mock, customers and orders are
  seeded demo data in SQLite, and the knowledge base is four short markdown files.
- **LLM behaviour is not systematically evaluated.** Tests use a scripted fake model, so they verify the
  graph, approval flow and error handling, not how well a real model classifies or plans. Manual runs showed
  the planner can misjudge a case (for example escalating a large refund itself), which prompt rules reduce but
  do not eliminate. Large refunds and shipped-order cancellations are approval-gated; the model never moves
  money directly.
- **Single process, local storage.** SQLite and an in-memory vector index suit a demo, not concurrent
  multi-instance deployment. There is no queue, rate limiting of API callers, or observability stack.
- **Free-tier constraints.** Groq's free tier is rate limited and its available models change; calls are
  retried with backoff, but a long demo can still be throttled and will then escalate.
- **Notifications.** A rejected or pending approval is returned in the API response only; the customer is
  not emailed.

## Notes
- Windows with Application Control may block compiled extensions. For SQLAlchemy, deleting the
  `*.pyd` files under `site-packages/sqlalchemy` makes it use the pure-Python fallback; for `ormsgpack`
  (used by LangGraph's checkpointer), install a version that isn't blocked (e.g. `pip install ormsgpack==1.13.0`).
- Model and thresholds are config (`.env`): `GROQ_MODEL`, `REFUND_APPROVAL_THRESHOLD`, `EMBEDDING_BACKEND`.
- Free-tier rate limits are the main constraint; a typical turn uses 2-4 LLM calls.