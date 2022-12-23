from django.urls import path
from . import views

urlpatterns = [
    path('register', views.register_account),
    path('account-details', views.get_account_details),
]