import datetime
import re
from pathlib import Path
import os
import json
from django.contrib.auth import get_user_model
from dotenv import load_dotenv
from rest_framework.decorators import api_view, permission_classes, authentication_classes, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import BasicAuthentication
from rest_framework_api_key.permissions import HasAPIKey

from .models import Tag, Tag_Class
from .serializers import UserSerializer, TagClassSerializer, LayerFileSerializer, LayerSerializer, \
    LayerParameterSerializer, LayerInputOutputSerializer, InputOutputChannelsSerializer, TagSerializer, \
    TagLayerSerializer
from rest_framework_api_key.models import APIKey

# Create your views here.

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)

j_val = {}
with open(os.getenv("FIELDS_PATH"), "r") as f:
    j_val.update(json.load(f))


def check_shapes(shape_json: list,
                 parameter_keys: list,
                 name: str):
    cp_all_keys = set(j_val["allowed_keywords"])
    cp_all_keys.update(set(parameter_keys))

    un_allowed = j_val["not_allowed_chars"]

    for idx, data_dict in enumerate(shape_json):
        var_set = set(data_dict["variables"])

        data = data_dict

        if len(var_set.difference(cp_all_keys)) != 0:
            data = Response(["Name {}'s, variable provided in channel {} aren't valid".format(name,
                                                                                              idx)], status=400)

        elif len(data_dict["shape_str"]) != len(re.sub(un_allowed, '', data_dict["shape_str"])):
            data = Response(["Name {}'s has a invalid Character in shape_str of channel {}".format(name,
                                                                                                   idx)], status=400)

        yield data, idx


def validate_parameter(parameter: dict,
                       parameter_name: str):
    def check_type(data_type, val):

        if type(val) is not eval(j_val["valid_datatype"][data_type]):
            return Response(["Value {} is not of the same type as {}, for param: {}".format(val,
                                                                                            j_val[
                                                                                                "valid_datatype"
                                                                                            ][data_type],
                                                                                            parameter_name)],
                            status=400)

    if parameter.get("options"):

        try:
            for i in parameter["options"]:
                resp = check_type(parameter["type"], i)
                if type(resp) == Response:
                    return resp

        except TypeError:
            return Response(["Options provided are invalid, for param: {}".format(parameter_name)],
                            status=400)

    dtype_keys = set(j_val["valid_datatype"].keys())

    if parameter["type"] not in dtype_keys:
        return Response(["type {} is not a valid type".format(parameter["type"],
                                                              parameter_name)],
                        status=400)

    if parameter.get("default_value"):
        resp = check_type(parameter["type"], parameter["default_value"]["default"])
        if type(resp) == Response:
            return resp

    return parameter


@api_view(["PATCH"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def grant_api_key(request):
    data = request.data
    user = str(request.user)
    model_data = get_user_model().objects.get(pk=user)

    if not model_data.is_staff:
        return Response({"invalid": "You are not a member of staff"}, status=400)

    model_data = get_user_model().objects.get(pk=data["req_user"])

    if model_data.api_key:
        return Response({"invalid": "User has already been assigned an apikey"}, status=400)

    api_key, key = APIKey.objects.create_key(name="{}-service-{}".format(data["req_user"],
                                                                         datetime.datetime.now()))

    serializer = UserSerializer(model_data,
                                data={"api_key": True},
                                partial=True)

    if serializer.is_valid(raise_exception=True):

        serializer.save()
        ret_data = {"key": key}

        return Response(ret_data, status=201)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["POST"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def insert_tag_class(request):
    data = request.data

    user = str(request.user)

    model_data = get_user_model().objects.get(pk=user)
    if not model_data.is_staff:
        return Response({"invalid": "You are not a member of staff"}, status=400)

    request.data["user_name"] = user
    request.data["tag_class_name"] = request.data["tag_class_name"].upper()
    serializer = TagClassSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):

        serializer.save()
        return Response(serializer.data, status=200)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["POST"])
@permission_classes([HasAPIKey])
@parser_classes([MultiPartParser, FormParser])
def insert_layer_file(request):
    data = request.data

    data = {"upload": data["file"]}

    ser = LayerFileSerializer(data=data)

    if ser.is_valid(raise_exception=True):
        ser.save()

        data = ser.data

        data.pop("upload")
        return Response(data, status=200)

    return Response({"Invalid": "Improper data was sent"}, status=400)


@api_view(["POST"])
@permission_classes([HasAPIKey])
def insert_layer(request):
    data = request.data

    layer_data = {
        "layer_name": data.pop("layer_name"),
        "layer_file_id": data.pop("layer_file_id")
    }

    lyr_ser = LayerSerializer(data=layer_data)

    if lyr_ser.is_valid(raise_exception=True):
        layer_id = lyr_ser.save().id

    p_list = []

    for d_k, d_v in data["parameter_dict"].items():
        d_v["param_name"] = d_k
        d_v["layer_id"] = layer_id

        ret = validate_parameter(d_v, d_k)
        if type(ret) == Response:
            return ret

        p_list.append(ret)

    lyr_ser = LayerParameterSerializer(data=p_list,
                                       many=True,
                                       allow_null=True)

    if lyr_ser.is_valid(raise_exception=True):
        lyr_ser.save()

    ii_list = ([{"is_input": True}] * len(data["input_shape"])) + ([{"is_input": False}] * len(data["output_shape"]))

    in_out = data.pop("input_shape") + data.pop("output_shape")

    for ii, io in (zip(ii_list, in_out)):

        input_output = {"input_name": io["name"],
                        "layer_id": layer_id}

        input_output.update(ii)

        liop_ser = LayerInputOutputSerializer(data=input_output)

        if liop_ser.is_valid(raise_exception=True):
            liop_id = liop_ser.save().id

        channel_list = []
        for c_ret in check_shapes(io["shape"],
                                  list(data["parameter_dict"].keys()),
                                  io["name"]):

            c_data, channel_num = c_ret

            if type(c_data) == Response:
                return c_data

            channel_list.append({"input_id": liop_id,
                                 "input_shape_str": c_data["shape_str"],
                                 "variables": c_data["variables"],
                                 "channel_number": channel_num,
                                 "operation": c_data.get("operation")})

        c_ser = InputOutputChannelsSerializer(data=channel_list,
                                              many=True)

        if c_ser.is_valid(raise_exception=True):
            c_ser.save()

    tag_list = data.pop("tags")

    for tag in tag_list:
        tag["tag_name"] = tag["tag_name"].lower()

        qs = Tag.objects.filter(pk=tag["tag_name"])

        if qs.exists():

            for t in qs:
                t.count += 1

                tag_class_name = t.tag_class_name.tag_class_name
                t.save()
        else:
            tag_ser = TagSerializer(data=tag)

            if tag_ser.is_valid(raise_exception=True):
                tag_class_name = tag_ser.save().tag_class_name.tag_class_name

            t_c = Tag_Class.objects.get(pk=tag_class_name)

            t_c.tag_count += 1

            t_c.save()

    for tag in tag_list:
        tag["layer_id"] = layer_id

    tag_lay_ser = TagLayerSerializer(data=tag_list,
                                     many=True)

    if tag_lay_ser.is_valid(raise_exception=True):
        tag_lay_ser.save()

    return Response(["The data was successfully stored"], status=200)
