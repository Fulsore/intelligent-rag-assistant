"""
chatbot/urls.py
"""

from django.urls import path
from .views import chat, session_stats, clear_session

urlpatterns = [
    path("chat/", chat),
    path("chat/stats/", session_stats),
    path("chat/clear/", clear_session),
]