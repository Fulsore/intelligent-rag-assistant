"""
knowledge/loaders/pdf_loader.py
Healthcare KIS - PDF Document Loader

Loads PDF files from the knowledge/data directory,
extracts text page by page, and returns LangChain Documents.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# We use pypdf (pure Python, no system deps)
try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError(
        "pypdf not installed. Run: pip install pypdf"
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class PDFLoader:
    """
    Loads all PDF files from knowledge/data/ directory.
    Returns a flat list of LangChain Documents (one per page).
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_all(self) -> List[Document]:
        """Load all PDFs in data_dir. Returns list of Documents."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Create it and add PDF files."
            )

        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[PDFLoader] No PDF files found in {self.data_dir}")
            return []

        print(f"[PDFLoader] Found {len(pdf_files)} PDF file(s)")
        all_docs = []

        for pdf_path in pdf_files:
            docs = self.load_file(pdf_path)
            all_docs.extend(docs)
            print(f"[PDFLoader] Loaded {len(docs)} pages from {pdf_path.name}")

        print(f"[PDFLoader] Total pages loaded: {len(all_docs)}")
        return all_docs

    def load_file(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF file. Returns one Document per page."""
        docs = []
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) < 50:  # skip nearly-empty pages
                    continue
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                    },
                )
                docs.append(doc)
        except Exception as e:
            print(f"[PDFLoader] Error reading {pdf_path.name}: {e}")

        return docs

    def load_txt_files(self) -> List[Document]:
        """Also loads plain .txt files from data_dir."""
        txt_files = list(self.data_dir.glob("*.txt"))
        docs = []
        for txt_path in txt_files:
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": txt_path.name, "page": 1},
                    ))
                    print(f"[PDFLoader] Loaded TXT: {txt_path.name}")
            except Exception as e:
                print(f"[PDFLoader] Error reading {txt_path.name}: {e}")
        return docs

    def load_all_files(self) -> List[Document]:
        """Load both PDFs and TXT files."""
        return self.load_all() + self.load_txt_files()


# Singleton
pdf_loader = PDFLoader()


if __name__ == "__main__":
    docs = pdf_loader.load_all_files()
    print(f"\nLoaded {len(docs)} total documents")
    if docs:
        print(f"Sample: {docs[0].page_content[:200]}")