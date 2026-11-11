# LLM Eval Harness: Design

Date: 2026-11-11

## Purpose

Measure the real Groq model's quality on the two LLM-driven decisions in the support graph:

- **Routing:** the supervisor's intent classification.
- **Planning:** the action proposed by `resolution_plan` / `order_change`.

The pytest suite already covers code correctness with scripted fakes. This harness covers model quality and is run on demand. It needs `GROQ_API_KEY` and the network, and is never part of `pytest`.

## Scope

In scope: a golden dataset, a runner that drives the real graph against the real model, scoring, and a report.

Out of scope: grading free-text replies, LLM-as-judge, RAG retrieval quality, and the approval/execution paths (covered by pytest).

## Layout

```
evals/
  __init__.py
  golden.jsonl     # ~40 cases
  scoring.py       # pure: score_case, aggregate (no network, unit-tested)
  run.py           # CLI + runner
tests/test_evals.py  # scoring + golden-file validation, no network
```

Evals live outside `tests/` so a plain `pytest` never calls the network.

## Golden case schema (one JSON object per line)

| field | type | notes |
|---|---|---|
| `id` | string | unique |
| `customer` | string | `C001`/`C002`/`C003`, a seeded customer |
| `message` | string | the latest customer message |
| `history` | list of `{role, content}` | optional prior turns, to test "that order" references |
| `expect_intent` | string | one of the graph's `INTENTS` |
| `expect_action` | string or null | `refund`, `complaint`, `escalate`, `cancel_order`, `change_address`, `none`, or `null` when the path proposes no action (knowledge, general, account_query) |
| `expect_order_id` | string or null | optional; checked only when present |

Coverage: every intent, boundary cases (for example "how do I cancel an order?" is a `policy_question`, not an `order_change`), and each customer's seeded orders (`ORD-1001`, `ORD-1002`, `ORD-2001`, `ORD-2002`, `ORD-3001`).

## Production change

`build_graph(checkpointer=None, interrupt_before=None)` passes `interrupt_before` to `compile()`. The argument is optional and backward-compatible. Nothing else in `app/` changes.

## Runner (`evals/run.py`)

1. Before importing `app`, set env vars to use a temp SQLite DB and temp checkpoint DB, and `EMBEDDING_BACKEND=tfidf` (same technique as `tests/conftest.py`). Then `init_db()` and `seed()`.
2. Build the real graph with `build_graph(MemorySaver(), interrupt_before=["approval_gate", "execute"])`.
3. For each case, invoke the graph with a fresh `thread_id`, the case's `customer_id`, and `history + message` as messages. Read `intent`, `proposed_action` and `errors` from the resulting state.
   - A path that proposes an action halts before `approval_gate` or `execute`, so nothing is executed. The DB and mock API are untouched.
   - A path that only answers runs to the end, and its action is `null`.
   - If the planner failed and `@safe_node` produced an escalation fallback, the case is counted as an **error** (failure), not an `escalate` action.
4. Cases run sequentially with a configurable delay (`--delay`, default 0.5s) to respect Groq rate limits. `invoke_llm` already retries on 429/5xx.

CLI: `python -m evals.run [--limit N] [--delay S] [--json PATH] [--min-intent X] [--min-action X]`

## Scoring (`evals/scoring.py`)

- **Routing correct:** result intent equals `expect_intent`.
- **Planning correct:** result action equals `expect_action` (both `null` counts as a match), and when `expect_order_id` is present the proposed `order_id` equals it. A case whose run errored is incorrect on both.
- Aggregates: overall and per-intent accuracy for each metric, an intent confusion matrix, the list of failing cases (id, message, expected vs got), and the error count.

## Report and exit code

Printed to stdout as a table plus failure details. `--json` writes the full per-case results for run-to-run comparison. Exit code is 0, or 1 if a `--min-intent` / `--min-action` threshold is set and missed (so it can gate CI later). A missing `GROQ_API_KEY` is a clear error and exit code 2.

## Testing

`tests/test_evals.py` (no network):
- `scoring` unit tests: match, mismatch, null-action, order_id check, error case, aggregation, confusion matrix.
- The golden file validates: unique ids, known customers, valid intents and actions, and each `expect_order_id` belongs to the case's customer.
- The runner is checked once with the `script_llm` fixture's fake LLM, to confirm it halts before `execute` and reads state correctly.
