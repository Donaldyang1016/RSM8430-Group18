"""
Evaluation harness for the L.O.V.E. agent.

Includes:
1) Retrieval metrics: Hit@k and MRR@k on a small golden set
2) Routing and safety accuracy checks
3) Build-plan flow regression check (multi-turn action)
4) Save / retrieve plan action checks (single-turn actions)
5) Edge-case / error-handling checks (empty input, gibberish, mid-flow switch)
6) Binary L.O.V.E.-style rubric checks
7) Failure taxonomy summary

Run:
    python evaluation/run_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.actions import (
    handle_build_plan,
    handle_retrieve_plan,
    handle_save_plan,
)
from agent.memory import SessionMemory
from agent.router import classify_intent
from agent.safety import screen_message
from state.store import create_session, init_db

try:
    from rag.retriever import get_retriever

    _HAS_RETRIEVER = True
    _RETRIEVER_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - fallback for missing local deps
    _HAS_RETRIEVER = False
    _RETRIEVER_IMPORT_ERROR = str(exc)

EVAL_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_CSV_PATH = EVAL_DIR / "results.csv"
SUMMARY_JSON_PATH = EVAL_DIR / "summary.json"


def _load_cases() -> dict[str, Any]:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rubric_checks(text: str) -> dict[str, int]:
    lower = text.lower()
    checks = {
        "listen_signal": int(
            bool(
                re.search(
                    r"(it sounds like|i hear you|what you're describing|it seems like)",
                    lower,
                )
            )
        ),
        "open_dialogue_question": int(bool("?" in text)),
        "validation_signal": int(
            bool(
                re.search(
                    r"(that sounds|it makes sense|anyone would feel|totally understandable)",
                    lower,
                )
            )
        ),
        "encouraging_step": int(
            bool(
                re.search(
                    r"(one step|could be|might help|would you like|feel doable)",
                    lower,
                )
            )
        ),
    }
    checks["overall_pass"] = int(all(v == 1 for v in checks.values()))
    return checks


def _fake_plan_llm(_system: str, _prompt: str) -> str:
    return (
        "Opening Statement: I'd like us to talk about how chores have been affecting us lately.\n"
        "Talking Points:\n"
        "1. I feel unheard when chores become blame-focused arguments.\n"
        "2. I want us to create a fair shared plan.\n"
        "3. I want to hear your view on what feels realistic.\n"
        "Validating Phrase: I know you have also been feeling stressed and stretched.\n"
        "Boundary Phrase: If either of us starts yelling, let's pause for ten minutes and restart calmly.\n"
        "Suggested Follow-up: What would a fair weekly plan look like for you?"
    )


def run_eval(k: int = 3) -> None:
    init_db()
    cases = _load_cases()
    results_rows: list[dict[str, Any]] = []
    taxonomy: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_cases = cases.get("retrieval", [])
    retrieval_hits = 0
    retrieval_mrr = 0.0

    if _HAS_RETRIEVER:
        retriever = get_retriever()
        for case in retrieval_cases:
            query = case["query"]
            expected_topics = set(case["expected_topics"])
            retrieved = retriever.retrieve(query, k=k)
            topics = [r.get("project_topic", "") for r in retrieved]

            rank = None
            for idx, topic in enumerate(topics, start=1):
                if topic in expected_topics:
                    rank = idx
                    break

            passed = rank is not None
            retrieval_hits += int(passed)
            retrieval_mrr += (1.0 / rank) if rank else 0.0
            if not passed:
                taxonomy["retrieval_miss"] += 1

            results_rows.append(
                {
                    "component": "retrieval",
                    "case_id": case["id"],
                    "pass": int(passed),
                    "details": json.dumps(
                        {
                            "expected_topics": sorted(list(expected_topics)),
                            "retrieved_topics": topics,
                            "rank": rank,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    else:
        taxonomy["retrieval_skipped_missing_dependencies"] += len(retrieval_cases)
        for case in retrieval_cases:
            results_rows.append(
                {
                    "component": "retrieval",
                    "case_id": case["id"],
                    "pass": 0,
                    "details": json.dumps(
                        {
                            "skipped": True,
                            "reason": _RETRIEVER_IMPORT_ERROR,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    routing_cases = cases.get("routing", [])
    routing_passes = 0
    for case in routing_cases:
        intent, _ = classify_intent(case["text"], action_state=None, llm_fn=None)
        passed = intent == case["expected_intent"]
        routing_passes += int(passed)
        if not passed:
            taxonomy["routing_mismatch"] += 1

        results_rows.append(
            {
                "component": "routing",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(
                    {"expected": case["expected_intent"], "actual": intent},
                    ensure_ascii=False,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    safety_cases = cases.get("safety", [])
    safety_passes = 0
    for case in safety_cases:
        verdict = screen_message(case["text"])
        actual = verdict["category"]
        passed = actual == case["expected_category"]
        safety_passes += int(passed)
        if not passed:
            taxonomy["safety_mismatch"] += 1

        results_rows.append(
            {
                "component": "safety",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(
                    {"expected": case["expected_category"], "actual": actual},
                    ensure_ascii=False,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Build-plan flow regression
    # ------------------------------------------------------------------
    plan_cases = cases.get("plan_flow", [])
    plan_passes = 0
    for case in plan_cases:
        session_id = create_session()
        memory = SessionMemory(session_id)
        messages = case["messages"]

        first = handle_build_plan(messages[0], memory, _fake_plan_llm)
        second = handle_build_plan(messages[1], memory, _fake_plan_llm)
        third = handle_build_plan(messages[2], memory, _fake_plan_llm)
        fourth = handle_build_plan(messages[3], memory, _fake_plan_llm)

        first_slot_issue = (first.get("slots") or {}).get("issue")
        final_issue = ((fourth.get("plan") or {}).get("slots") or {}).get("issue")

        passed = (
            first.get("type") == "follow_up"
            and not first_slot_issue
            and second.get("type") == "follow_up"
            and third.get("type") == "follow_up"
            and fourth.get("type") == "plan"
            and final_issue == messages[1]
        )
        plan_passes += int(passed)
        if not passed:
            taxonomy["plan_flow_regression"] += 1

        results_rows.append(
            {
                "component": "plan_flow",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(
                    {
                        "first_type": first.get("type"),
                        "first_slot_issue": first_slot_issue,
                        "final_type": fourth.get("type"),
                        "final_issue": final_issue,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Single-turn actions: save_plan, retrieve_plan
    # ------------------------------------------------------------------
    action_cases = cases.get("actions", [])
    action_passes = 0
    for case in action_cases:
        # Each action case runs in its own fresh session so prior runs don't
        # leak into the result.
        session_id = create_session()
        memory = SessionMemory(session_id)

        # 1. Confirm the router still classifies the trigger text as expected.
        intent, _ = classify_intent(case["intent_text"], action_state=None, llm_fn=None)
        intent_ok = intent == case["expected_intent"]

        # 2. Exercise the action handler. If the test requires a pre-existing
        #    plan (e.g. save / retrieve), we seed one using the same fake LLM
        #    used by the plan_flow block — keeps the eval deterministic and
        #    free of network calls.
        action_ok = False
        details: dict[str, Any] = {
            "expected_intent": case["expected_intent"],
            "actual_intent": intent,
        }

        seed_plan: dict[str, Any] | None = None
        if case.get("requires_existing_plan"):
            seed_messages = [
                "help me build a conversation plan",
                "i feel unheard when we argue about chores",
                "i want us to agree on a fair plan and stop blaming each other",
                "gentle but direct",
            ]
            for msg in seed_messages:
                result = handle_build_plan(msg, memory, _fake_plan_llm)
            seed_plan = result.get("plan") if result.get("type") == "plan" else None

        if case["expected_intent"] == "save_plan":
            outcome = handle_save_plan(session_id, seed_plan)
            action_ok = bool(outcome.get("success"))
            details["action_success"] = outcome.get("success")
            details["action_message"] = outcome.get("message", "")[:120]

        elif case["expected_intent"] == "retrieve_plan":
            # Save first so retrieval has something to find
            if seed_plan is not None:
                handle_save_plan(session_id, seed_plan)
            outcome = handle_retrieve_plan(session_id)
            message = outcome.get("message", "")
            expected_substrings = case.get("expected_response_contains", [])
            substr_hits = all(s.lower() in message.lower() for s in expected_substrings)
            action_ok = bool(outcome.get("success")) and substr_hits
            details["action_success"] = outcome.get("success")
            details["substring_hits"] = substr_hits
            details["plans_returned"] = len(outcome.get("plans", []))

        passed = intent_ok and action_ok
        action_passes += int(passed)
        if not passed:
            if not intent_ok:
                taxonomy["action_routing_mismatch"] += 1
            if not action_ok:
                taxonomy[f"action_handler_failed_{case['expected_intent']}"] += 1

        results_rows.append(
            {
                "component": "actions",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(details, ensure_ascii=False),
            }
        )

    # ------------------------------------------------------------------
    # Edge cases: empty input, gibberish, mid-flow topic switch
    # ------------------------------------------------------------------
    edge_cases = cases.get("edge_cases", [])
    edge_passes = 0
    for case in edge_cases:
        details: dict[str, Any] = {"description": case.get("description", "")}
        passed = False
        try:
            # Two shapes: single-text (empty / gibberish) or multi-message (switch)
            if "messages" in case:
                # Multi-turn topic switch: simulate a build_plan start, then
                # check the router respects an explicit pivot to another intent.
                session_id = create_session()
                memory = SessionMemory(session_id)
                final_intent: str | None = None
                for msg in case["messages"]:
                    state = memory.get_action_state()
                    final_intent, _ = classify_intent(msg, action_state=state, llm_fn=None)
                    # Mirror what the app would do: if intent is build_plan,
                    # run the handler so action_state gets created.
                    if final_intent == "build_plan":
                        handle_build_plan(msg, memory, _fake_plan_llm)
                expected = case["expected_final_intent"]
                passed = final_intent == expected
                details["expected_final_intent"] = expected
                details["actual_final_intent"] = final_intent

            else:
                text = case.get("text", "")
                # Empty input is screened by safety
                if "expected_category" in case:
                    verdict = screen_message(text)
                    passed = (
                        verdict["category"] == case["expected_category"]
                        and verdict["safe"] == case.get("expected_safe", False)
                    )
                    details["expected_category"] = case["expected_category"]
                    details["actual_category"] = verdict["category"]
                    details["actual_safe"] = verdict["safe"]
                # Gibberish: should route without raising
                elif "expected_intent" in case:
                    intent, _ = classify_intent(text, action_state=None, llm_fn=None)
                    passed = intent == case["expected_intent"]
                    details["expected_intent"] = case["expected_intent"]
                    details["actual_intent"] = intent
        except Exception as exc:
            details["exception"] = f"{type(exc).__name__}: {exc}"
            passed = False
            taxonomy["edge_case_exception"] += 1

        edge_passes += int(passed)
        if not passed and "exception" not in details:
            taxonomy["edge_case_mismatch"] += 1

        results_rows.append(
            {
                "component": "edge_cases",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(details, ensure_ascii=False),
            }
        )

    # ------------------------------------------------------------------
    # Binary rubric checks
    # ------------------------------------------------------------------
    rubric_cases = cases.get("rubric_samples", [])
    rubric_passes = 0
    for case in rubric_cases:
        checks = _rubric_checks(case["text"])
        passed = checks["overall_pass"] == 1
        rubric_passes += int(passed)
        if not passed:
            for key, value in checks.items():
                if key != "overall_pass" and value == 0:
                    taxonomy[f"rubric_missing_{key}"] += 1

        results_rows.append(
            {
                "component": "rubric",
                "case_id": case["id"],
                "pass": int(passed),
                "details": json.dumps(checks, ensure_ascii=False),
            }
        )

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "case_id", "pass", "details"])
        writer.writeheader()
        writer.writerows(results_rows)

    retrieval_total = max(len(retrieval_cases), 1)
    routing_total = max(len(routing_cases), 1)
    safety_total = max(len(safety_cases), 1)
    plan_total = max(len(plan_cases), 1)
    action_total = max(len(action_cases), 1)
    edge_total = max(len(edge_cases), 1)
    rubric_total = max(len(rubric_cases), 1)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "retrieval_hit_at_k": (
                round(retrieval_hits / retrieval_total, 4) if _HAS_RETRIEVER else None
            ),
            "retrieval_mrr_at_k": (
                round(retrieval_mrr / retrieval_total, 4) if _HAS_RETRIEVER else None
            ),
            "routing_accuracy": round(routing_passes / routing_total, 4),
            "safety_accuracy": round(safety_passes / safety_total, 4),
            "plan_flow_pass_rate": round(plan_passes / plan_total, 4),
            "action_pass_rate": round(action_passes / action_total, 4),
            "edge_case_pass_rate": round(edge_passes / edge_total, 4),
            "rubric_pass_rate": round(rubric_passes / rubric_total, 4),
        },
        "counts": {
            "total_cases": len(results_rows),
            "passed_cases": int(sum(r["pass"] for r in results_rows)),
            "failed_cases": int(len(results_rows) - sum(r["pass"] for r in results_rows)),
        },
        "failure_taxonomy": dict(taxonomy),
        "artifacts": {
            "results_csv": str(RESULTS_CSV_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
        },
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Evaluation complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOVE agent evaluation suite.")
    parser.add_argument("--k", type=int, default=3, help="Top-k for retrieval evaluation.")
    args = parser.parse_args()
    run_eval(k=args.k)


if __name__ == "__main__":
    main()
