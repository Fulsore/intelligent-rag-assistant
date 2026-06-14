# Data Sources

## 1. Clinical Reference PDF
- Source: https://www.oyschst.edu.ng/elib/dashboard/ebooks/jYRS8aEg.pdf
- Purpose: Base medical knowledge extraction (diseases, clinical descriptions)

## 2. Structured Medical Dataset (Custom)
- File: knowledge/data/medical_kb.txt
- Includes: Diabetes, Hypertension, Malaria
- Format: Structured disease → symptoms → causes → treatment

## 3. Embedding Model
- sentence-transformers/all-MiniLM-L6-v2

## 4. Vector Database
- FAISS (local vector store used for semantic retrieval)