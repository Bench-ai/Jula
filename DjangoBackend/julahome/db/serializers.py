from rest_framework import serializers
from .models import Layer, Tag_Class, Tag, Tag_Layer_Model, Layer_File, Layer_Parameter, Layer_Input_Output, \
    Input_Output_Channels
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


class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ["id",
                  "layer_name",
                  "layer_file_id",
                  "creation_timestamp"]

    def create(self, data):
        layer = Layer.objects.create(layer_name=data["layer_name"],
                                     layer_file_id=data["layer_file_id"])

        return layer

    def update(self, instance, validated_data):
        instance.layer_name = validated_data.get('layer_name', instance.layer_name)
        instance.layer_file_id = validated_data.get('layer_file_id', instance.layer_file_id)
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
                  "default_value",
                  "type",
                  "options",
                  "creation_timestamp",
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

        return Layer.objects.create(**data)

    def update(self, instance, validated_data):
        instance.layer_file_id = validated_data.get('layer_file_id', instance.layer_file_id)
        instance.param_name = validated_data.get('param_name', instance.param_name)
        instance.description = validated_data.get('description', instance.description)
        instance.default_value = validated_data.get('default_value', instance.default_value)
        instance.type = validated_data.get('type', instance.type)
        instance.options = validated_data.get('options', instance.options)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.save()


class LayerInputOutputSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer_Input_Output
        fields = ["id",
                  "layer_id",
                  "input_name",
                  "is_input",
                  "is_deleted",
                  "creation_timestamp"]

    def create(self, data):
        # lip = Layer_Input_Output.objects.create(layer_id=data["layer_id"],
        #                                         input_name=data["input_name"],
        #                                         is_input=data["is_input"],
        #                                         is_deleted=data["is_deleted"])

        return Layer_Input_Output.objects.create(**data)

    def update(self, instance, validated_data):
        instance.input_name = validated_data.get('input_name', instance.input_name)
        instance.is_input = validated_data.get('is_input', instance.is_input)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.save()


class InputOutputChannelsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Input_Output_Channels
        fields = ["id",
                  "input_id",
                  "input_shape_str",
                  "variables",
                  "channel_number",
                  "operation",
                  "is_deleted",
                  "creation_timestamp"]

    def create(self, data):
        # ioc = Input_Output_Channels.objects.create(input_id=data["input_id"],
        #                                            input_shape_str=data["input_shape_str"],
        #                                            variables=data["variables"],
        #                                            channel_number=data["channel_number"],
        #                                            operation=data["operation"],
        #                                            is_deleted=data["is_deleted"])

        return Input_Output_Channels.objects.create(**data)

    def update(self, instance, validated_data):
        instance.input_shape_str = validated_data.get('input_shape_str', instance.input_shape_str)
        instance.variables = validated_data.get('variables', instance.variables)
        instance.channel_number = validated_data.get('channel_number', instance.channel_number)
        instance.operation = validated_data.get('operation', instance.operation)
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
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
        tag_class = Tag_Class.objects.create(tag_name=data["tag_name"],
                                             tag_class_name=data["tag_class_name"])

        return tag_class


class TagLayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag_Layer_Model

        fields = [
            "tag_name",
            "layer_id"
            "creation_timestamp",
            "is_deleted"
        ]

    def create(self, data):
        tag_layer = Tag_Class.objects.create(tag_name=data["tag_name"],
                                             layer_id=data["layer_id"])

        return tag_layer

    def update(self, instance, validated_data):
        instance.is_deleted = validated_data.get('is_deleted', instance.is_deleted)
        instance.save()

        return instance
