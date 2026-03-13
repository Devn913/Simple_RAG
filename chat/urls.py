from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('ask/', views.ask_question, name='ask_question'),
    path('delete/<int:doc_id>/', views.delete_document, name='delete_document'),
]
