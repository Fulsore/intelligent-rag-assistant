"""
scoring/centrality.py
Healthcare KIS - Centrality Scoring

Ranks retrieved chunks by their "centrality" to the question.
Most central chunks are used first in context building.
"""

from typing import List, Tuple
from langchain_core.documents import Document


class CentralityScorer:
    """
    Ranks retrieved documents by how central/relevant they are to the query.
    Uses keyword hit density + position-weighted scoring.
    """

    def rank_by_centrality(
        self,
        question: str,
        docs: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Returns list of (doc, centrality_score) sorted descending.
        """
        q_keywords = self._keywords(question)
        if not q_keywords:
            return [(doc, 1.0) for doc in docs]

        scored = []
        for doc in docs:
            score = self._score_doc(doc.page_content, q_keywords)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_doc(self, text: str, keywords: List[str]) -> float:
        words = text.lower().split()
        total_words = max(len(words), 1)

        # Count keyword hits
        hits = sum(text.lower().count(kw) for kw in keywords)

        # Density: hits per 100 words
        density = (hits / total_words) * 100

        # Early-position bonus: keywords in first 100 words weighted 2x
        early_text = " ".join(words[:100])
        early_hits = sum(early_text.count(kw) for kw in keywords)
        early_bonus = early_hits * 0.5

        # Unique keyword coverage
        unique_hits = sum(1 for kw in keywords if kw in text.lower())
        coverage = unique_hits / len(keywords)

        score = (0.4 * density) + (0.3 * early_bonus) + (0.3 * coverage * 10)
        return round(min(score, 100.0), 3)

    def _keywords(self, text: str) -> List[str]:
        stopwords = {
            "a","an","the","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","can","what","which","who","how","and",
            "but","or","for","to","of","in","on","at","by","from","with",
            "about","that","this","these","those","me","my","we","you","he",
            "she","it","they","them","tell","explain","describe","give","list",
        }
        words = text.lower().split()
        return [
            w.strip(".,!?;:'\"()") for w in words
            if len(w) > 3 and w.strip(".,!?;:'\"()") not in stopwords
        ]


# Singleton
centrality_scorer = CentralityScorer()