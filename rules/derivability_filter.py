"""
rules/derivability_filter.py
Healthcare KIS - Derivability Pre-Filter

Before scoring the full answer, quickly filters out context chunks
that cannot contribute to a derivable response.
"""

from typing import List, Tuple
from langchain_core.documents import Document


class DerivabilityFilter:
    """
    Pre-filters context documents to retain only those that
    are likely to contribute to a derivable answer.
    """

    def __init__(self, min_keyword_hits: int = 1):
        self.min_keyword_hits = min_keyword_hits

    def filter(
        self,
        question: str,
        docs: List[Document],
    ) -> Tuple[List[Document], List[Document]]:
        """
        Returns (retained_docs, filtered_out_docs)
        """
        q_keywords = self._keywords(question)
        if not q_keywords:
            return docs, []

        retained = []
        filtered_out = []

        for doc in docs:
            text_lower = doc.page_content.lower()
            hits = sum(1 for kw in q_keywords if kw in text_lower)
            if hits >= self.min_keyword_hits:
                retained.append(doc)
            else:
                filtered_out.append(doc)

        # Always keep at least 2 docs even if none pass the filter
        if len(retained) < 2 and docs:
            retained = docs[:2]
            filtered_out = docs[2:]

        return retained, filtered_out

    def _keywords(self, text: str) -> List[str]:
        stopwords = {
            "a","an","the","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","can","what","which","who","how","and",
            "but","or","for","to","of","in","on","at","by","from","with",
            "about","that","this","these","those","tell","explain","give",
            "list","describe","define","what","me","my","i","we","you",
        }
        words = text.lower().split()
        return [
            w.strip(".,!?;:'\"()")
            for w in words
            if len(w) > 3 and w.strip(".,!?;:'\"()") not in stopwords
        ]


# Singleton
derivability_filter = DerivabilityFilter()