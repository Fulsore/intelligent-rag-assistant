"""
rules/threshold_rule.py
Healthcare KIS - Threshold Rules

Defines score thresholds used by the rule engine and pipeline
to decide whether to answer, abstain, or warn.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdConfig:
    """
    All configurable thresholds in the KIS pipeline.
    Adjust these to tune strictness.
    """

    # Derivability: below this → "cannot reliably answer from context"
    min_derivability: float = 0.35

    # Trust score (0-100): below this → show low-confidence warning
    min_trust_score: float = 30.0

    # Retrieval similarity: below this → discard the chunk
    min_retrieval_similarity: float = 0.25

    # Max chunks to use in context
    max_context_chunks: int = 5

    # Min chunk quality (words)
    min_chunk_words: int = 20

    # Trust score buckets for UI display
    high_trust_threshold: float = 80.0
    medium_trust_threshold: float = 55.0
    low_trust_threshold: float = 30.0

    def get_trust_label(self, score: float) -> str:
        if score >= self.high_trust_threshold:
            return "High"
        elif score >= self.medium_trust_threshold:
            return "Medium"
        elif score >= self.low_trust_threshold:
            return "Low"
        else:
            return "Very Low"

    def should_abstain(self, derivability: float, trust: float) -> bool:
        """
        Returns True if the system should refuse to answer
        rather than provide a low-quality response.
        """
        return derivability < self.min_derivability and trust < self.min_trust_score

    def get_abstain_message(self) -> str:
        return (
            "I could not find sufficient information in the knowledge base "
            "to answer this question reliably. Please consult a qualified "
            "healthcare professional or refer to official medical literature."
        )


# Default singleton
thresholds = ThresholdConfig()