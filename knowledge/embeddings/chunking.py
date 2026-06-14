"""
knowledge/embeddings/chunking.py
Healthcare KIS - Text Chunking

Splits loaded Documents into smaller chunks for embedding.
Configurable chunk size and overlap.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MedicalChunker:
    """
    Splits documents into chunks optimized for medical Q&A retrieval.
    Uses RecursiveCharacterTextSplitter with medical-aware separators.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Separators ordered by preference — paragraph → sentence → word
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents into chunks."""
        chunks = self._splitter.split_documents(documents)

        # Post-filter: remove chunks that are too short to be useful
        chunks = [
            c for c in chunks
            if len(c.page_content.split()) >= 15
        ]

        print(f"[Chunker] {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """Split raw text string into chunks."""
        return self._splitter.split_text(text)

    def get_stats(self, chunks: List[Document]) -> dict:
        """Return basic stats about the chunk set."""
        if not chunks:
            return {"total": 0}
        lengths = [len(c.page_content.split()) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_words": round(sum(lengths) / len(lengths), 1),
            "min_words": min(lengths),
            "max_words": max(lengths),
        }


# Singleton with default settings
medical_chunker = MedicalChunker(chunk_size=500, chunk_overlap=60)


if __name__ == "__main__":
    # Quick test
    from langchain_core.documents import Document
    test_docs = [
        Document(page_content="Hypertension is a common condition. " * 20)
    ]
    chunks = medical_chunker.split(test_docs)
    print(f"Chunks: {len(chunks)}")
    print(f"Stats: {medical_chunker.get_stats(chunks)}")