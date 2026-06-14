"""
scoring/derivability.py
Healthcare KIS - Derivability Scoring Module

Scores how derivable (inferable) an answer is from the retrieved context.
Higher score = answer is strongly grounded in retrieved documents.
"""

from typing import List, Tuple
from langchain_core.documents import Document


class DerivabilityScorer:
    """
    Scores how well an answer can be derived from the given context chunks.
    Uses keyword overlap + semantic position + chunk coverage.
    """

    def __init__(self, min_threshold: float = 0.35):
        self.min_threshold = min_threshold

    def score(
        self,
        question: str,
        answer: str,
        context_docs: List[Document]
    ) -> Tuple[float, dict]:
        """
        Returns (derivability_score: float, breakdown: dict)
        score is between 0.0 and 1.0
        """
        if not context_docs or not answer:
            return 0.0, {"error": "No context or answer provided"}

        q_keywords = self._extract_keywords(question)
        a_keywords = self._extract_keywords(answer)

        context_text = " ".join(
            doc.page_content.lower() for doc in context_docs
        )

        # Metric 1: How many answer keywords appear in context
        a_in_context = [w for w in a_keywords if w in context_text]
        keyword_coverage = (
            len(a_in_context) / len(a_keywords) if a_keywords else 0.0
        )

        # Metric 2: How many question keywords are satisfied by context
        q_in_context = [w for w in q_keywords if w in context_text]
        question_coverage = (
            len(q_in_context) / len(q_keywords) if q_keywords else 0.0
        )

        # Metric 3: Chunk utilization (how many chunks contributed)
        contributing_chunks = 0
        for doc in context_docs:
            doc_lower = doc.page_content.lower()
            hits = sum(1 for w in a_keywords if w in doc_lower)
            if hits >= 2:
                contributing_chunks += 1

        chunk_util = min(
            contributing_chunks / max(len(context_docs), 1), 1.0
        )

        # Metric 4: Answer length vs context length ratio (penalize hallucination)
        context_len = len(context_text.split())
        answer_len = len(answer.split())
        length_ratio = min(context_len / max(answer_len * 2, 1), 1.0)

        # Weighted final score
        score = (
            0.40 * keyword_coverage +
            0.25 * question_coverage +
            0.20 * chunk_util +
            0.15 * length_ratio
        )

        breakdown = {
            "keyword_coverage": round(keyword_coverage, 3),
            "question_coverage": round(question_coverage, 3),
            "chunk_utilization": round(chunk_util, 3),
            "length_ratio": round(length_ratio, 3),
            "contributing_chunks": contributing_chunks,
            "total_chunks": len(context_docs),
            "final_score": round(score, 3),
        }

        return round(score, 3), breakdown

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords, removing stopwords."""
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "need", "must", "what", "which", "who", "whom", "whose",
            "when", "where", "why", "how", "and", "but", "or", "nor",
            "for", "so", "yet", "both", "either", "neither", "not",
            "of", "in", "on", "at", "by", "from", "with", "about",
            "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "up", "down", "out", "off",
            "over", "under", "again", "further", "then", "once", "that",
            "this", "these", "those", "i", "me", "my", "myself", "we",
            "our", "you", "your", "he", "she", "it", "they", "them",
        }
        words = text.lower().split()
        keywords = [
            w.strip(".,!?;:'\"()[]{}") for w in words
            if len(w) > 3 and w not in stopwords
        ]
        return list(set(keywords))

    def is_derivable(self, score: float) -> bool:
        return score >= self.min_threshold


# Singleton
derivability_scorer = DerivabilityScorer()