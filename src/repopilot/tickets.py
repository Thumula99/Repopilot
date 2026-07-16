from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PriorityResult:
    priority: str
    reason: str
    score: int


CRITICAL_TERMS = [
    "payment",
    "billing",
    "security",
    "data loss",
    "breach",
    "production down",
    "all users",
    "cannot login",
    "login down",
    "outage",
    "crash",
]
HIGH_TERMS = ["auth", "authentication", "error", "exception", "failed", "timeout", "api", "database", "db"]
LOW_TERMS = ["typo", "color", "spacing", "copy", "text", "alignment", "minor", "cosmetic"]


def classify_priority(text: str) -> PriorityResult:
    """Rule-based priority classifier for a free MVP.

    Later you can replace this with a small trained PyTorch/sklearn model and keep
    the same interface.
    """
    lowered = text.lower()
    score = 0
    reasons = []

    for term in CRITICAL_TERMS:
        if term in lowered:
            score += 3
            reasons.append(f"critical signal: {term}")
    for term in HIGH_TERMS:
        if term in lowered:
            score += 2
            reasons.append(f"high-impact signal: {term}")
    for term in LOW_TERMS:
        if term in lowered:
            score -= 1
            reasons.append(f"low-impact signal: {term}")

    if re.search(r"\b(500|503|401|403)\b", lowered):
        score += 2
        reasons.append("HTTP error code signal")
    if re.search(r"\b(all|every|many|multiple)\s+(user|users|customer|customers)\b", lowered):
        score += 3
        reasons.append("many users affected")

    if score >= 6:
        priority = "Critical"
    elif score >= 3:
        priority = "High"
    elif score >= 1:
        priority = "Medium"
    else:
        priority = "Low"

    reason = "; ".join(reasons[:5]) or "No strong risk signals found."
    return PriorityResult(priority=priority, reason=reason, score=score)


def build_basic_issue(problem: str) -> str:
    priority = classify_priority(problem)
    title = problem.strip().split("\n")[0][:90] or "Bug report"
    return f"""Title: {title}

Summary:
{problem.strip()}

Priority:
{priority.priority} — {priority.reason}

Steps to reproduce:
1. Open the affected feature.
2. Perform the action described above.
3. Observe the failure.

Expected behavior:
The feature should complete successfully without errors.

Actual behavior:
{problem.strip()}

Suggested labels:
bug, needs-triage, priority-{priority.priority.lower()}

Extra information needed:
- Environment: local/staging/production
- Browser/device or API client
- Error logs or screenshots
- Reproduction frequency
"""
