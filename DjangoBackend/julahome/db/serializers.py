from rest_framework import serializers
from .models import Layer


class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ["layer_name",
                  "upload",
                  "uid",
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
