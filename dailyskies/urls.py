from django.urls import path
from . import views

urlpatterns = [
    path('', views.weatherview, name='home'),  # Map root path to your view
]
