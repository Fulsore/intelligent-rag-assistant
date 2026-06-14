"""
rules/rule_engine.py
Healthcare KIS - Rule-Based Context Filter

Applies domain rules to filter, flag, or adjust context
before it reaches the LLM. Ensures only safe, relevant
content is passed through.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


# Keywords that trigger warnings or filtering
FLAGGED_KEYWORDS = [
    "suicide", "self-harm", "overdose", "euthanasia",
    "illegal drug", "controlled substance", "prescription fraud",
    "self-medicate", "bypass doctor",
]

# Keywords that must be present for medical context (basic domain check)
MEDICAL_DOMAIN_KEYWORDS = [
    "patient", "treatment", "diagnosis", "symptom", "disease",
    "medication", "therapy", "clinical", "medical", "health",
    "drug", "dose", "condition", "surgery", "hospital", "doctor",
    "nurse", "prescription", "chronic", "acute", "infection",
]

OUT_OF_SCOPE_TOPICS = [
    "cryptocurrency", "stock market", "sports", "cooking recipe",
    "weather", "politics", "election", "celebrity",
]


class RuleEngine:
    """
    Applies rule-based filtering and validation to:
    1. Detect flagged / unsafe content
    2. Check medical domain relevance
    3. Filter out irrelevant context chunks
    """

    def __init__(
        self,
        min_medical_hits: int = 1,
        block_flagged: bool = False,  # warn but don't block by default
    ):
        self.min_medical_hits = min_medical_hits
        self.block_flagged = block_flagged

    def evaluate(
        self,
        question: str,
        context_docs: List[Document],
    ) -> Dict[str, Any]:
        """
        Runs all rules against the question and context.

        Returns a result dict:
        {
            passed: bool,
            flagged_keywords: List[str],
            out_of_scope: bool,
            medical_relevance: bool,
            filtered_docs: List[Document],
            warnings: List[str],
            rule_notes: str,
        }
        """
        q_lower = question.lower()
        warnings = []
        flagged = []

        # Rule 1: Flag sensitive keywords in question
        for kw in FLAGGED_KEYWORDS:
            if kw in q_lower:
                flagged.append(kw)

        if flagged:
            warnings.append(
                f"Sensitive topic detected: {', '.join(flagged)}. "
                "Answering with care and medical disclaimer."
            )

        # Rule 2: Out-of-scope check
        out_of_scope = any(topic in q_lower for topic in OUT_OF_SCOPE_TOPICS)
        if out_of_scope:
            warnings.append(
                "Question appears outside the medical/healthcare domain."
            )

        # Rule 3: Medical domain relevance in context
        all_context = " ".join(
            doc.page_content.lower() for doc in context_docs
        )
        medical_hits = sum(
            1 for kw in MEDICAL_DOMAIN_KEYWORDS if kw in all_context
        )
        medical_relevance = medical_hits >= self.min_medical_hits

        if not medical_relevance and context_docs:
            warnings.append(
                "Retrieved context has low medical domain relevance."
            )

        # Rule 4: Filter context chunks — remove very short or irrelevant ones
        filtered_docs = self._filter_context(question, context_docs)

        # Overall pass/fail
        passed = (
            not out_of_scope
            and medical_relevance
            and (not self.block_flagged or not flagged)
        )

        return {
            "passed": passed,
            "flagged_keywords": flagged,
            "out_of_scope": out_of_scope,
            "medical_relevance": medical_relevance,
            "medical_domain_hits": medical_hits,
            "filtered_docs": filtered_docs,
            "original_doc_count": len(context_docs),
            "filtered_doc_count": len(filtered_docs),
            "warnings": warnings,
            "rule_notes": "; ".join(warnings) if warnings else "All rules passed.",
        }

    def _filter_context(
        self,
        question: str,
        docs: List[Document],
    ) -> List[Document]:
        """
        Remove chunks that are too short or have zero relevance to question.
        Always keep at least 2 chunks.
        """
        q_words = set(question.lower().split())
        scored = []

        for doc in docs:
            text = doc.page_content
            word_count = len(text.split())

            # Skip very short chunks
            if word_count < 15:
                continue

            # Score by word overlap with question
            doc_words = set(text.lower().split())
            overlap = len(q_words & doc_words)
            scored.append((doc, overlap))

        # Sort by overlap descending
        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [doc for doc, _ in scored]

        # Ensure minimum 2 docs
        if len(filtered) < 2 and docs:
            filtered = docs[:2]

        return filtered

    def get_safe_answer_prefix(self, flagged_keywords: List[str]) -> str:
        """Returns a prefix warning for flagged questions."""
        if not flagged_keywords:
            return ""
        return (
            "⚠️ This question involves sensitive medical topics. "
            "The following answer is for informational purposes only. "
            "Please seek immediate professional help if needed.\n\n"
        )


# Singleton
rule_engine = RuleEngine()