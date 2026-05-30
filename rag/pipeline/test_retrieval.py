from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "rag/vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

query = input("Ask Question: ")

results = db.similarity_search(
    query,
    k=3
)

print("\nTOP RESULTS\n")

for i, result in enumerate(results):

    print(f"\nResult {i+1}\n")

    print(result.page_content)