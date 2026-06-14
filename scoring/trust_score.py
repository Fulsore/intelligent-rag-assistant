"""
scoring/trust_score.py
Healthcare KIS - Trust Score Aggregator

Combines multiple scoring signals into a single trust score.
Returns a normalized 0-100 score displayed to users as confidence level.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document


class TrustScoreCalculator:
    """
    Aggregates derivability + retrieval confidence + context coherence
    into a 0-100 trust score shown in the UI.
    """

    def calculate(
        self,
        question: str,
        answer: str,
        context_docs: List[Document],
        retrieval_scores: List[float],
        derivability_score: float,
        derivability_breakdown: Dict[str, Any],
        rule_filter_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns full trust report dict with overall score and components.
        """

        # Component 1: Derivability (0-1 → 0-100)
        deriv_component = derivability_score * 100

        # Component 2: Retrieval confidence (avg cosine similarity 0-1 → 0-100)
        if retrieval_scores:
            avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
            retrieval_component = min(avg_retrieval * 100, 100)
        else:
            retrieval_component = 0.0

        # Component 3: Context coherence — how consistent are the chunks
        coherence_component = self._calc_coherence(context_docs)

        # Component 4: Rule compliance bonus/penalty
        rule_bonus = 10.0 if rule_filter_result.get("passed") else -10.0
        if rule_filter_result.get("flagged_keywords"):
            rule_bonus -= 5.0 * len(rule_filter_result["flagged_keywords"])

        # Weighted aggregate
        base_score = (
            0.45 * deriv_component +
            0.30 * retrieval_component +
            0.25 * coherence_component
        )

        final_score = max(0.0, min(100.0, base_score + rule_bonus))

        # Confidence label
        if final_score >= 80:
            confidence_label = "High"
            confidence_color = "green"
        elif final_score >= 55:
            confidence_label = "Medium"
            confidence_color = "amber"
        elif final_score >= 30:
            confidence_label = "Low"
            confidence_color = "orange"
        else:
            confidence_label = "Very Low"
            confidence_color = "red"

        return {
            "trust_score": round(final_score, 1),
            "confidence_label": confidence_label,
            "confidence_color": confidence_color,
            "components": {
                "derivability": round(deriv_component, 1),
                "retrieval_confidence": round(retrieval_component, 1),
                "context_coherence": round(coherence_component, 1),
                "rule_adjustment": rule_bonus,
            },
            "derivability_breakdown": derivability_breakdown,
            "sources_used": len(context_docs),
            "rule_filter": rule_filter_result,
        }

    def _calc_coherence(self, docs: List[Document]) -> float:
        """
        Estimate coherence by checking keyword overlap across chunks.
        More overlap → higher coherence → more consistent context.
        """
        if len(docs) < 2:
            return 70.0  # single chunk, neutral score

        def keywords(text):
            words = set(text.lower().split())
            return {w.strip(".,!?;:'\"") for w in words if len(w) > 4}

        kw_sets = [keywords(doc.page_content) for doc in docs]
        pairwise_overlaps = []

        for i in range(len(kw_sets)):
            for j in range(i + 1, len(kw_sets)):
                a, b = kw_sets[i], kw_sets[j]
                if not a or not b:
                    continue
                jaccard = len(a & b) / len(a | b)
                pairwise_overlaps.append(jaccard)

        if not pairwise_overlaps:
            return 50.0

        avg_jaccard = sum(pairwise_overlaps) / len(pairwise_overlaps)
        # Map jaccard (typically 0.05-0.4) to 0-100
        coherence = min(avg_jaccard * 300, 100.0)
        return round(coherence, 1)


# Singleton
trust_calculator = TrustScoreCalculator()