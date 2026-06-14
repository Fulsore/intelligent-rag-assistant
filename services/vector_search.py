"""
services/vector_search.py
Healthcare KIS - Vector Search Service

Handles loading the FAISS vectorstore and performing similarity searches.
Returns top-k documents with their similarity scores.
"""

import os
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorSearchService:
    """
    Manages the FAISS vector store and provides similarity search.
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTORSTORE_PATH = "rag/vectorstore"

    def __init__(self):
        self._vectorstore: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def _load(self):
        """Lazy-load the vectorstore on first use."""
        if self._vectorstore is None:
            print("[VectorSearch] Loading embedding model...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[VectorSearch] Loading FAISS vectorstore...")
            self._vectorstore = FAISS.load_local(
                self.VECTORSTORE_PATH,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            print("[VectorSearch] Vectorstore ready.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Returns list of (Document, similarity_score) tuples.
        Scores are cosine similarity 0-1 (higher = more similar).
        """
        self._load()

        # FAISS with_score returns (doc, score) — lower L2 is better
        # We use normalized embeddings so L2 distance ≈ 2 * (1 - cosine)
        results_with_scores = self._vectorstore.similarity_search_with_score(
            query, k=top_k
        )

        # Convert L2 distance to cosine similarity: sim = 1 - (dist/2)
        converted = []
        for doc, l2_dist in results_with_scores:
            cosine_sim = max(0.0, 1.0 - (l2_dist / 2.0))
            if cosine_sim >= score_threshold:
                converted.append((doc, round(cosine_sim, 4)))
                print("\n" + "=" * 80)
                print("QUERY:", query)
                print("=" * 80)

                for i, (doc, score) in enumerate(converted):
                    print(f"\nRESULT {i+1}")
                    print("SCORE:", score)

                    print("CONTENT:")
                    print(doc.page_content[:1000])

                    print("-" * 80)

        return converted

    def search_docs_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """Returns only documents (no scores)."""
        results = self.search(query, top_k=top_k)
        return [doc for doc, _ in results]

    def get_scores_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[float]:
        """Returns only similarity scores."""
        results = self.search(query, top_k=top_k)
        return [score for _, score in results]


# Singleton — vectorstore loaded once at startup
vector_search = VectorSearchService()