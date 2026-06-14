# import pandas as pd

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS

# from langchain_core.documents import Document


# print("Loading cleaned data...")

# df = pd.read_csv(
#     "rag/cleaned_data/final_cleaned_data.csv"
# )

# documents = []

# for text in df['text']:

#     documents.append(
#         Document(page_content=text)
#     )

# print(f"Loaded {len(documents)} documents")


# print("Splitting documents...")

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# chunks = splitter.split_documents(documents)

# print(f"Created {len(chunks)} chunks")


# print("Loading embedding model...")

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# print("Creating vector database...")

# vectorstore = FAISS.from_documents(
#     chunks,
#     embedding_model
# )

# vectorstore.save_local(
#     "rag/vectorstore"
# )

# print("FAISS vector database created")












# New
from knowledge.loaders.pdf_loader import pdf_loader
from knowledge.embeddings.chunking import medical_chunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading documents...")

# STEP 1: Load PDF + TXT
docs = pdf_loader.load_all_files()

print(f"Loaded documents: {len(docs)}")

if len(docs) == 0:
    raise ValueError("No documents loaded. Check knowledge/data folder.")

# STEP 2: Chunking (YOU ALREADY BUILT THIS RIGHT)
chunks = medical_chunker.split(docs)

print(f"Chunks created: {len(chunks)}")

# STEP 3: Embeddings
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 4: Build FAISS
print("Creating FAISS index...")

vectorstore = FAISS.from_documents(chunks, embedding_model)

# STEP 5: SAVE INSIDE KNOWLEDGE FOLDER (IMPORTANT)
SAVE_PATH = "knowledge/vectorstore"

vectorstore.save_local(SAVE_PATH)

print(f"Vectorstore saved at {SAVE_PATH}")
print("DONE 🚀")