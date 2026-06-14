"""
scoring/score_store.py
Healthcare KIS - Score Store

Keeps the last N query score reports in memory for session analytics.
"""

from collections import deque
from typing import Dict, Any, List
from datetime import datetime


class ScoreStore:
    """
    In-memory store for recent query trust reports.
    Useful for session-level analytics and debugging.
    """

    def __init__(self, max_history: int = 50):
        self._history: deque = deque(maxlen=max_history)

    def add(self, question: str, trust_report: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            **trust_report,
        }
        self._history.appendleft(entry)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return list(self._history)[:n]

    def get_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {"total_queries": 0}

        scores = [entry["trust_score"] for entry in self._history]
        avg = sum(scores) / len(scores)
        return {
            "total_queries": len(self._history),
            "avg_trust_score": round(avg, 1),
            "max_trust_score": round(max(scores), 1),
            "min_trust_score": round(min(scores), 1),
        }

    def clear(self) -> None:
        self._history.clear()


# Singleton
score_store = ScoreStore()