"""Evaluation of the RAG pipeline.

Each test case in the JSON file looks like one of these:
  Answerable:   {"question": "...", "expected_source": "file.pdf", "expected_keywords": ["18|eighteen"]}
  Not in docs:  {"question": "...", "answerable": false}
"""
import json
import time
from pathlib import Path

from app.rag import NO_ANSWER, answer_question


PAUSE_SECONDS = 4  # pause between questions so the free tier is not overloaded


def _ask(question: str) -> dict:
    """Ask one question; wait and retry if the free tier says 'too many requests'."""
    for attempt in range(4):
        try:
            return answer_question(question)
        except Exception as e:
            msg = str(e).lower()
            if attempt < 3 and any(w in msg for w in ("429", "quota", "rate", "resource_exhausted")):
                time.sleep(20 * (attempt + 1))
            else:
                raise


def run_eval(eval_file: Path) -> dict:
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    ans_total = ans_retrieved = ans_correct = 0
    ref_total = ref_correct = 0
    latencies, details = [], []

    for case in cases:
        out = _ask(case["question"])
        time.sleep(PAUSE_SECONDS)
        latencies.append(out["timings"]["total_ms"])
        answer_text = out["answer"].lower()

        if case.get("answerable", True):
            ans_total += 1
            retrieved = case["expected_source"] in {s["file"] for s in out["sources"]}
            # a keyword like "18|eighteen" passes if ANY of the alternatives appears
            correct = all(
                any(alt.strip().lower() in answer_text for alt in k.split("|"))
                for k in case["expected_keywords"]
            )
            ans_retrieved += retrieved
            ans_correct += correct
            details.append({"question": case["question"], "type": "answerable",
                            "retrieval_hit": retrieved, "answer_correct": correct})
        else:
            ref_total += 1
            refused = NO_ANSWER.lower() in answer_text or "could not find" in answer_text
            ref_correct += refused
            details.append({"question": case["question"], "type": "not_in_docs",
                            "correctly_refused": refused})

    def rate(a, b):
        return round(a / b, 3) if b else None

    return {
        "total_cases": len(cases),
        "answerable_cases": ans_total,
        "not_in_docs_cases": ref_total,
        "retrieval_hit_rate": rate(ans_retrieved, ans_total),
        "answer_accuracy": rate(ans_correct, ans_total),
        "correct_refusal_rate": rate(ref_correct, ref_total),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "details": details,
    }
