from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List

from .store import ChromaStore


@dataclass
class EvalCase:
    question: str
    expected_keyword: str


@dataclass
class EvalResult:
    question: str
    expected_keyword: str
    passed: bool
    latency_seconds: float
    top_source: str


def parse_eval_cases(raw: str) -> list[EvalCase]:
    """Parse lines like: question | expected_keyword."""
    cases: list[EvalCase] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        question, expected = [part.strip() for part in line.split("|", 1)]
        if question and expected:
            cases.append(EvalCase(question=question, expected_keyword=expected))
    return cases


def run_retrieval_eval(store: ChromaStore, cases: Iterable[EvalCase], k: int = 5) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        start = time.perf_counter()
        hits = store.search(case.question, k=k)
        latency = time.perf_counter() - start
        combined = "\n".join(hit.text for hit in hits).lower()
        passed = case.expected_keyword.lower() in combined
        top_source = ""
        if hits:
            top_source = hits[0].metadata.get("path") or hits[0].metadata.get("title") or hits[0].metadata.get("url", "")
        results.append(
            EvalResult(
                question=case.question,
                expected_keyword=case.expected_keyword,
                passed=passed,
                latency_seconds=latency,
                top_source=top_source,
            )
        )
    return results


def summarize_eval(results: List[EvalResult]) -> dict:
    if not results:
        return {"total": 0, "passed": 0, "accuracy": 0.0, "avg_latency_seconds": 0.0}
    passed = sum(1 for result in results if result.passed)
    avg_latency = sum(result.latency_seconds for result in results) / len(results)
    return {
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results),
        "avg_latency_seconds": avg_latency,
    }
