from django.urls import path
from . import views

urlpatterns = [
    path('grant-key', views.grant_api_key),
    path('insert-tag-class', views.insert_tag_class),
    path('insert-layer-file', views.insert_layer_file),
    path('insert-layer', views.insert_layer),
    path('insert-json-file', views.insert_json_file),
    path('insert-tag', views.insert_tag),
    path('insert-layer-tag', views.insert_layer_tag),
]
