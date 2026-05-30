from rest_framework.decorators import api_view

from rest_framework.response import Response

from rag.pipeline.rag_chatbot import chatbot


@api_view(["POST"])
def chat(request):

    question = request.data.get("question")

    answer = chatbot(question)

    return Response({
        "question": question,
        "answer": answer
    })