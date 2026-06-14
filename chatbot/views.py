"""
chatbot/views.py
Healthcare KIS - Chat API Views
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from services.rag_pipeline import run_rag_pipeline
from scoring.score_store import score_store


@api_view(["POST"])
def chat(request):
    """
    POST /chat/
    Body: { "question": "..." }
    Returns full answer + trust score + derivability + rule results
    """
    question = request.data.get("question", "").strip()

    if not question:
        return Response(
            {"error": "Question is required"},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        result = run_rag_pipeline(question)
        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {
                "question": question,
                "answer": "An error occurred processing your question. Please try again.",
                "error": str(e),
                "trust": {"score": 0, "label": "Error", "color": "red"},
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
def session_stats(request):
    """
    GET /chat/stats/
    Returns session-level analytics from ScoreStore.
    """
    stats = score_store.get_stats()
    recent = score_store.get_recent(n=5)
    return Response({
        "stats": stats,
        "recent_queries": recent,
    })


@api_view(["DELETE"])
def clear_session(request):
    """
    DELETE /chat/clear/
    Clears the in-memory score store.
    """
    score_store.clear()
    return Response({"message": "Session cleared."})