from django.apps import AppConfig
from .views import load_faiss_index, load_vectors_from_db


class DocumentQnaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "document_qna"
