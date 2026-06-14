# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS

# from rag.pipeline.llm import query_llm


# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db = FAISS.load_local(
#     "rag/vectorstore",
#     embedding_model,
#     allow_dangerous_deserialization=True
# )


# def chatbot(question):

#     results = db.similarity_search(
#         question,
#         k=3
#     )

#     context = "\n".join([
#         result.page_content
#         for result in results
#     ])

#     prompt = f"""
#     You are a hyperlocal commerce assistant.

#     Answer ONLY from provided context.

#     Context:
#     {context}

#     Question:
#     {question}
#     """

#     answer = context

#     return answer