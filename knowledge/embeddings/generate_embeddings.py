"""
knowledge/embeddings/generate_embeddings.py
Healthcare KIS - Embedding Generation Script

Run this once (or whenever you update PDFs) to:
1. Load all PDFs/TXTs from knowledge/data/
2. Chunk them
3. Embed with HuggingFace sentence-transformers
4. Save FAISS vectorstore to rag/vectorstore/

Usage (from project root):
    python knowledge/embeddings/generate_embeddings.py
"""

import sys
import os
from pathlib import Path

# Make sure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.loaders.pdf_loader import PDFLoader
from knowledge.embeddings.chunking import MedicalChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_PATH = PROJECT_ROOT / "rag" / "vectorstore"
DATA_DIR = PROJECT_ROOT / "knowledge" / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate():
    print("=" * 50)
    print("Healthcare KIS - Vectorstore Generator")
    print("=" * 50)

    # Step 1: Load documents
    print("\n[1/4] Loading documents from knowledge/data/ ...")
    loader = PDFLoader(data_dir=DATA_DIR)
    docs = loader.load_all_files()

    if not docs:
        print(
            "\n❌ No documents found!\n"
            f"Add PDF or TXT files to: {DATA_DIR}\n"
            "Then re-run this script."
        )
        sys.exit(1)

    print(f"✅ Loaded {len(docs)} document pages")

    # Step 2: Chunk
    print("\n[2/4] Chunking documents...")
    chunker = MedicalChunker(chunk_size=500, chunk_overlap=60)
    chunks = chunker.split(docs)
    stats = chunker.get_stats(chunks)
    print(f"✅ Created {stats['total_chunks']} chunks")
    print(f"   Avg words/chunk: {stats['avg_words']}")

    # Step 3: Load embedding model
    print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL} ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embedding model ready")

    # Step 4: Create and save FAISS vectorstore
    print("\n[4/4] Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))

    print(f"✅ Vectorstore saved to: {VECTORSTORE_PATH}")
    print("\n🎉 Done! Your knowledge base is ready.")
    print(f"   {len(chunks)} chunks indexed from {len(docs)} pages")


if __name__ == "__main__":
    generate()