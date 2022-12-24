from rest_framework import serializers
from .models import Layer, Tag_Class, Tag, TagLayerModel
from django.contrib.auth import get_user_model


class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ["layer_name",
                  "upload",
                  "layer_parameters"]

    def create(self, data):
        layer = Layer.create(layer_name=data["layer_name"],
                             upload=data["upload"],
                             uid=data["uid"],
                             layer_parameters=data["layer_parameters"])

        return layer

    def update(self, instance, validated_data):
        instance.layer_name = validated_data.get('layer_name', instance.layer_name)
        instance.upload = validated_data.get('upload', instance.upload)
        instance.layer_parameters = validated_data.get('layer_parameters', instance.layer_parameters)
        instance.save()

        return instance


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ["user_name",
                  "email",
                  "password",
                  "api_key"]

    def create(self, data):
        user = get_user_model().objects.create_user(user_name=data["user_name"],
                                                    email=data["email"],
                                                    password=data["password"])

        return user

    def update(self, instance, validated_data):
        instance.email = validated_data.get('email', instance.email)
        instance.api_key = validated_data.get('api_key', instance.api_key)
        instance.save()

        return instance


class TagClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag_Class

        fields = [
            "tag_class_name",
            "user_name",
            "tag_count"
        ]

    def create(self, data):
        tag_class = Tag_Class.objects.create(tag_class_name=data["tag_class_name"],
                                             user_name=data["user_name"])

        return tag_class


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag_Class

        fields = [
            "tag_name",
            "user_name",
            "tag_class_name"
        ]

    def create(self, data):
        tag_class = Tag_Class.objects.create(tag_class_name=data["tag_class_name"],
                                             user_name=data["user_name"])

        return tag_class
