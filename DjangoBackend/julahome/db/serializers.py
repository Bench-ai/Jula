from rest_framework import serializers
from .models import Layer, Tag_Class, Tag, Tag_Layer_Model, Layer_File, Layer_Parameter, Layer_Input_Output, \
    Input_Output_Channels, Json_File, input_output_details
from django.contrib.auth import get_user_model


class LayerFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer_File
        fields = ["id",
                  "upload",
                  "creation_timestamp",
                  "download_count"]

    def create(self, data):
        layer = Layer_File.objects.create(upload=data["upload"])

        return layer

    def update(self, instance, validated_data):
        instance.upload = validated_data.get('upload', instance.upload)
        instance.download_count = validated_data.get('download_count', instance.download_count)
        instance.save()

        return instance


class JsonFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Json_File
        fields = ["id",
                  "upload",
                  "creation_timestamp"]

    def create(self, data):
        j = Json_File.objects.create(upload=data["upload"])

        return j

    def update(self, instance, validated_data):
        instance.upload = validated_data.get('upload', instance.upload)
        instance.save()

        return instance


class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ["id",
                  "layer_name",
                  "layer_file_id",
                  "layer_json_id",
                  "forward_list",
                  "forward_dict",
                  "creation_timestamp",
                  "default"]

    def create(self, data):
        layer = Layer.objects.create(layer_name=data["layer_name"],
                                     layer_file_id=data["layer_file_id"],
                                     layer_json_id=data["layer_json_id"],
                                     forward_list=data["forward_list"],
                                     forward_dict=data["forward_dict"],
                                     default=data["default"])

        return layer

    def update(self, instance, validated_data):
        instance.layer_name = validated_data.get('layer_name', instance.layer_name)
        instance.layer_file_id = validated_data.get('layer_file_id', instance.layer_file_id)
        instance.layer_file_id = validated_data.get('layer_json_id', instance.layer_json_id)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.forward_list = validated_data.get('forward_list', instance.forward_list)
        instance.forward_dict = validated_data.get('forward_dict', instance.forward_dict)
        instance.default = validated_data.get('default', instance.default)
        instance.save()

        return instance


class InputOutputSerializer(serializers.ModelSerializer):
    class Meta:
        model = input_output_details
        fields = ["id",
                  "layer_id",
                  "name",
                  "description",
                  "is_output",
                  "creation_timestamp"]

    def create(self, data):
        input_out = input_output_details.objects.create(layer_id=data["layer_id"],
                                                        name=data["name"],
                                                        description=data["description"],
                                                        is_output=data["is_output"])

        return input_out

    def update(self, instance, validated_data):
        instance.name = validated_data.get('name', instance.name)
        instance.description = validated_data.get('description', instance.description)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.save()

        return instance


class LayerParameterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer_Parameter
        fields = ["id",
                  "layer_id",
                  "param_name",
                  "description",
                  "type",
                  "default_and_options",
                  "creation_timestamp",
                  "is_forward",
                  "is_deleted"]

    def create(self, data):
        # layer = Layer.objects.create(layer_id=data["layer_id"],
        #                              layer_file_id=data["layer_file_id"],
        #                              param_name=data["param_name"],
        #                              description=data["description"],
        #                              default_value=data["default_value"],
        #                              type=data["type"],
        #                              options=data["options"],
        #                              is_deleted=data["is_deleted"])

        return Layer_Parameter.objects.create(**data)

    def update(self, instance, validated_data):
        instance.layer_file_id = validated_data.get('layer_file_id', instance.layer_file_id)
        instance.param_name = validated_data.get('param_name', instance.param_name)
        instance.description = validated_data.get('description', instance.description)
        instance.type = validated_data.get('type', instance.type)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.layer_file_id = validated_data.get('layer_file_id', instance.layer_file_id)
        instance.default_and_options = validated_data.get('default_and_options', instance.default_and_options)
        instance.save()


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
            "tag_count",
            "creation_timestamp"
        ]

    def create(self, data):
        tag_class = Tag_Class.objects.create(tag_class_name=data["tag_class_name"],
                                             user_name=data["user_name"])

        return tag_class


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag

        fields = [
            "tag_name",
            "tag_class_name",
            "creation_timestamp",
            "count"
        ]

    def create(self, data):
        tag_class = Tag.objects.create(tag_name=data["tag_name"],
                                       tag_class_name=data["tag_class_name"])

        return tag_class


class TagLayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag_Layer_Model

        fields = [
            "tag_name",
            "layer_id",
            "creation_timestamp",
            "is_deleted"
        ]

    def create(self, data):
        tag_layer = Tag_Layer_Model.objects.create(tag_name=data["tag_name"],
                                                   layer_id=data["layer_id"])

        return tag_layer

    def update(self, instance, validated_data):
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.save()

        return instance
