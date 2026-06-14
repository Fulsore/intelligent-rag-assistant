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