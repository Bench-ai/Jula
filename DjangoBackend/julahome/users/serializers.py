from rest_framework import serializers
from django.contrib.auth import get_user_model


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ["user_name",
                  "email",
                  "password"]

    def create(self, data):
        user = get_user_model().objects.create_user(user_name=data["user_name"],
                                                    email=data["email"],
                                                    password=data["password"])

        return user


class AccountDataSerializer(serializers.ModelSerializer):

    class Meta:
        model = get_user_model()
        fields = ["user_name",
                  "email",
                  "is_staff",
                  "creation_timestamp"]

