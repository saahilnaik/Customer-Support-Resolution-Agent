# LLM Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A script that runs a golden dataset through the real support graph with the real Groq model and reports routing (intent) and planning (action) accuracy, without executing any action.

**Architecture:** The runner builds the real graph with `interrupt_before=["approval_gate", "execute"]` and a `MemorySaver`, so every action-proposing path halts with `proposed_action` in state and nothing is refunded or ticketed. Pure scoring functions are separated from the I/O runner so they are unit-tested with no network.

**Tech Stack:** Python 3.14, LangGraph (`MemorySaver`), pytest, argparse. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-11-llm-eval-harness-design.md`

## Global Constraints

- Evals live in a top-level `evals/` package, never in `tests/`; plain `pytest` must not touch the network.
- Only production change: `build_graph(checkpointer=None, interrupt_before=None)`, backward-compatible.
- Runner sets `DATABASE_URL`, `CHECKPOINT_DB` (temp dir) and `EMBEDDING_BACKEND=tfidf` **before** importing `app`; it must NOT override `GROQ_API_KEY` (it comes from `.env`).
- Interrupt points are exactly `["approval_gate", "execute"]`.
- Valid `expect_action`: `refund`, `complaint`, `escalate`, `cancel_order`, `change_address`, `none`, or `null` (path proposes no action).
- Valid `expect_intent`: `policy_question`, `account_query`, `complaint`, `refund_request`, `order_change`, `other`.
- Seeded customers `C001`/`C002`/`C003`; orders `ORD-1001`, `ORD-1002` (C001), `ORD-2001`, `ORD-2002` (C002), `ORD-3001` (C003).
- CLI: `python -m evals.run [--limit N] [--delay S] [--json PATH] [--min-intent X] [--min-action X]`; `--delay` default 0.5.
- Exit codes: 0 ok, 1 a `--min-*` threshold missed, 2 missing `GROQ_API_KEY`.
- A run where `@safe_node` caught an error (state `errors` non-empty), or where the graph raised, is an **error** and counts as incorrect on both metrics.
- Run tests with `.venv/Scripts/python -m pytest -q` (Git Bash).

## Review Focus

- Groq exhausts retries or the graph raises mid-run: that case is recorded as an error and the remaining cases still run (Task 4).
- `GROQ_API_KEY` missing: clear message, exit 2, before any case runs; and `.env`'s key is not clobbered by env setup (Task 4).
- Zero cases scored (`--limit 0`) or every case errored: no ZeroDivisionError, accuracy reported as 0.0 (Task 2).
- Malformed or blank lines in `golden.jsonl`: loader fails with the line number, blank lines are skipped (Task 3).
- `--json PATH` whose parent directory doesn't exist: the directory is created, not a crash (Task 4).

---

### Task 1: `interrupt_before` on `build_graph`

**Files:**
- Modify: `app/graph/builder.py` (`build_graph`, last lines)
- Test: `tests/test_graph_api.py` (append one test)

**Interfaces:**
- Produces: `build_graph(checkpointer=None, interrupt_before: list[str] | None = None)`; passes `interrupt_before` straight to `g.compile(...)`.

- [ ] **Step 1: Write the failing test `test_interrupt_before_halts_before_execute`** in `tests/test_graph_api.py`. Use `build_graph(MemorySaver(), interrupt_before=["approval_gate", "execute"])` and the `script_llm` fixture with an `order_change` flow (supervisor JSON `{"intent":"order_change",...}`, then order-change JSON cancelling `ORD-3001`; see the `change()` helper). Invoke with `{"messages": [("user", "cancel ORD-3001")], "customer_id": "C003"}` and a `thread_id` config. Assert `graph.get_state(cfg).values["proposed_action"]["action"] == "cancel_order"`, `graph.get_state(cfg).next == ("approval_gate",)`, and that the `ORD-3001` order status in the DB is still not `cancelled`.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_graph_api.py::test_interrupt_before_halts_before_execute -q`. Expected: FAIL (`unexpected keyword argument 'interrupt_before'`).
- [ ] **Step 3: Implement** the optional parameter in `build_graph` and pass it to `compile`.
- [ ] **Step 4: Run** the new test, then the full suite `.venv/Scripts/python -m pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: optional interrupt_before on build_graph`.

### Task 2: Scoring (pure)

**Files:**
- Create: `evals/__init__.py` (empty), `evals/scoring.py`
- Test: `tests/test_evals.py`

**Interfaces:**
- Produces:
  - Result dict (one per case, built by the runner): `{"id": str, "intent": str | None, "action": str | None, "order_id": str | None, "errored": bool, "error": str | None}`.
  - `score_case(case: dict, result: dict) -> dict` returns `{"intent_ok": bool, "action_ok": bool}`. `action_ok` requires `result["action"] == case["expect_action"]` (both `None` matches), plus `order_id` equality when `case` has a non-null `expect_order_id`. An errored result is `False` on both.
  - `aggregate(cases: list[dict], results: list[dict]) -> dict` returns `{"total", "errors", "intent_accuracy", "action_accuracy", "per_intent": {intent: {"n", "intent_accuracy", "action_accuracy"}}, "confusion": {expected_intent: {got_intent: count}}, "failures": [{"id", "message", "expect_intent", "got_intent", "expect_action", "got_action", "error"}]}`. Per-intent grouping is by `expect_intent`. Accuracies are floats in `[0, 1]`, `0.0` when `total == 0`.

- [ ] **Step 1: Write failing tests** in `tests/test_evals.py`: `test_score_case_match`, `test_score_case_intent_wrong_action_right`, `test_score_case_null_action_matches_none_result`, `test_score_case_order_id_checked_only_when_expected`, `test_score_case_errored_is_wrong_on_both`, `test_aggregate_accuracy_confusion_and_failures`, `test_aggregate_empty_returns_zero_accuracy_without_error` (Review Focus: zero cases), `test_aggregate_all_errored_is_zero`.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_evals.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `score_case` and `aggregate` in `evals/scoring.py` per the Interfaces block (plain dicts, no classes).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: eval scoring`.

### Task 3: Golden dataset and loader

**Files:**
- Create: `evals/golden.jsonl`, `evals/dataset.py`
- Test: `tests/test_evals.py` (append)

**Interfaces:**
- Produces: `load_golden(path: Path = GOLDEN_PATH) -> list[dict]` (`GOLDEN_PATH` = `evals/golden.jsonl`). Skips blank lines; on invalid JSON raises `ValueError(f"{path}:{lineno}: ...")`. Each case has `id, customer, message, expect_intent, expect_action` and optional `history` (`[{"role": "user"|"assistant", "content": str}]`) and `expect_order_id`. Also `validate_case(case: dict) -> list[str]` returning problems (empty if valid) using the Global Constraints value sets plus an ownership map `ORDER_OWNER = {"ORD-1001": "C001", "ORD-1002": "C001", "ORD-2001": "C002", "ORD-2002": "C002", "ORD-3001": "C003"}`.

- [ ] **Step 1: Write failing tests**: `test_golden_file_is_valid` (every case passes `validate_case`, ids unique, at least 35 cases, every intent and every non-null action value appears at least twice), `test_load_golden_skips_blank_lines`, `test_load_golden_reports_line_number_on_bad_json` (writes a temp file with a bad line 2 and asserts the message contains `:2:`), `test_validate_case_rejects_order_of_another_customer`.
- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** `evals/dataset.py`.
- [ ] **Step 4: Write `evals/golden.jsonl`**, about 40 cases. Distribution to cover:
  - `policy_question` (about 7, `expect_action` null): returns, shipping times, warranty, "how do I cancel an order?" (boundary), "can I change my address after shipping?" (boundary), a question the KB cannot answer.
  - `account_query` (about 6, null): where is `ORD-1002`, list my orders, my account tier, each for the owning customer.
  - `refund_request` (about 8): damaged `ORD-1001` -> `refund`; refund for a delivered item `ORD-2001` -> `refund`; vague "I want my money back" without an order -> `none`; a refund asked inside a multi-turn `history`.
  - `complaint` (about 6): defective product -> `complaint`; late delivery -> `complaint`; "let me talk to a human" -> `escalate`.
  - `order_change` (about 9): cancel `ORD-3001` -> `cancel_order` (order_id `ORD-3001`); address change for `ORD-3001` with a full address -> `change_address`; cancel shipped `ORD-1002` -> `cancel_order`; cancel delivered `ORD-1001` -> `none` (ineligible); address change without an address -> `none`; cancel "that order" with `history` naming `ORD-3001`; cancel without any order id -> `none`.
  - `other` (about 4, null): greeting, thanks, off-topic.
  Mark expectations that are genuinely ambiguous sparingly; prefer cases the policy text and seeded data settle.
- [ ] **Step 5: Run** `.venv/Scripts/python -m pytest tests/test_evals.py -q`. Expected: PASS.
- [ ] **Step 6: Commit** `feat: eval golden dataset and loader`.

### Task 4: Runner and report

**Files:**
- Create: `evals/run.py`
- Test: `tests/test_evals.py` (append)

**Interfaces:**
- Consumes: `load_golden`, `score_case`, `aggregate`, and `build_graph(..., interrupt_before=...)`.
- Produces in `evals/run.py`:
  - `setup_env() -> None`: `os.environ.setdefault` for `DATABASE_URL`, `CHECKPOINT_DB` (temp dir) and `EMBEDDING_BACKEND="tfidf"`. Never touches `GROQ_API_KEY`. Called at the start of `main()` before any `app` import.
  - `build_eval_graph()`: imports `app` lazily, runs `init_db()` and `seed()`, returns `build_graph(MemorySaver(), interrupt_before=["approval_gate", "execute"])`.
  - `run_case(graph, case: dict) -> dict`: fresh `thread_id` per case; messages are `case["history"]` mapped to `(role, content)` tuples plus `("user", message)`; invokes the graph inside `try/except Exception` and returns the result dict from Task 2. `action` is `None` when state has no `proposed_action`, else `proposed_action["action"]`; `order_id` comes from `proposed_action["order_id"]` if present else `None`. `errored` is True when state `errors` is non-empty or the graph raised (`error` holds the text).
  - `format_report(summary: dict) -> str`: overall routing and planning accuracy and error count, a per-intent table, the confusion matrix, then each failure (id, message, expected vs got).
  - `main(argv: list[str] | None = None) -> int`: parses the CLI, returns the exit code per Global Constraints; `if __name__ == "__main__": sys.exit(main())`. Missing `GROQ_API_KEY` (read via `get_settings().groq_api_key`) prints a clear message and returns 2 before running any case. `--json PATH` creates the parent directory and writes `{"summary": ..., "results": [...]}`. Sleeps `--delay` seconds between cases.

- [ ] **Step 1: Write failing tests** using the `script_llm` fixture (the runner must use `llm_mod.get_llm()` like the app does, so the monkeypatch applies):
  - `test_run_case_halts_before_execute_and_reads_action`: scripted supervisor + order-change JSON for cancelling `ORD-3001`; assert result `intent == "order_change"`, `action == "cancel_order"`, `order_id == "ORD-3001"`, `errored is False`, and the order is not cancelled in the DB.
  - `test_run_case_answer_only_path_has_null_action`: scripted supervisor `other` then a general reply; assert `action is None`.
  - `test_run_case_records_graph_exception_as_error` (Review Focus): fake LLM script raises `RuntimeError` from the supervisor; assert `errored is True`, `error` non-empty, no exception escapes.
  - `test_run_case_safe_node_fallback_is_error`: planner returns invalid JSON action; assert `errored is True`.
  - `test_main_exits_2_without_groq_key` (Review Focus): monkeypatch `get_settings().groq_api_key` to `""`; `main(["--limit", "1"]) == 2`.
  - `test_setup_env_does_not_override_groq_key`: set `GROQ_API_KEY=abc`, call `setup_env()`, assert it is still `abc`.
  - `test_main_json_output_creates_parent_dir_and_threshold_exit_code` (Review Focus): patch `run_case` to return canned results; run `main(["--json", str(tmp/"a"/"b"/"out.json"), "--delay", "0", "--min-action", "1.0", "--limit", "2"])`; assert the file exists and the return code is 1 when the canned results miss the threshold.
- [ ] **Step 2: Run** the new tests. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `evals/run.py` per the Interfaces block. Use `graph.invoke(...)` then `graph.get_state(cfg).values` to read the halted state; use `time.sleep` for the delay (patchable).
- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest -q`. Expected: whole suite PASS.
- [ ] **Step 5: Commit** `feat: eval runner and report`.

### Task 5: Real-model run and docs

**Files:**
- Modify: `CLAUDE.md` (Commands block: add the eval command; Architecture: one short paragraph noting `evals/`, that it needs `GROQ_API_KEY`, and that it is not part of pytest), `README.md` (short "Evaluating the model" section)

- [ ] **Step 1: Smoke run** `.venv/Scripts/python -m evals.run --limit 5 --delay 1`. Expected: a report prints, exit code 0 or 1, no traceback, and `data/support.db` is unchanged (`git status` clean for `data/`; the runner used temp DBs).
- [ ] **Step 2: Full run** `.venv/Scripts/python -m evals.run --json evals/last_run.json`. Record the accuracy numbers. Investigate any case that is an **error** (as opposed to a wrong answer): an error is a harness or app bug, not model quality. Fix golden expectations only when the expectation itself was wrong, and say so in the commit message.
- [ ] **Step 3: Add `evals/last_run.json` to `.gitignore`** and update `CLAUDE.md` / `README.md`.
- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `docs: eval harness usage`.
