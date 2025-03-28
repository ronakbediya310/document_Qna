from django.urls import path
from . import views
from .views import query_api
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'), 
    path('index/', views.index, name='index'),
    path("api/query", query_api, name="query_api"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
