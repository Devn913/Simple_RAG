from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('ask/', views.ask_question, name='ask_question'),
    path('delete/<int:doc_id>/', views.delete_document, name='delete_document'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('validate-key/', views.validate_api_key, name='validate_api_key'),
    path('clear-keys/', views.clear_keys, name='clear_keys'),
]
