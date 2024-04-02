from django.contrib import admin
from django.urls import path
from . import views,utils

app_name = 'dateapp'

urlpatterns = [
    path("", utils.show_text_from_image, name= "show_text_from_image"),
]