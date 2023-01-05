import datetime
from pprint import pprint
from uuid import uuid4
import typing
from pathlib import Path
import os
import json
from django.contrib.auth import get_user_model
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from rest_framework.decorators import api_view, permission_classes, authentication_classes, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import BasicAuthentication
from rest_framework_api_key.permissions import HasAPIKey

from .models import Tag, Tag_Class, Layer, Tag_Layer_Model
from .serializers import UserSerializer, TagClassSerializer, LayerFileSerializer, LayerSerializer, \
    LayerParameterSerializer, TagSerializer, TagLayerSerializer, JsonFileSerializer, InputOutputSerializer
from rest_framework_api_key.models import APIKey

# Create your views here.

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)

j_val = {}
with open(os.getenv("FIELDS_PATH"), "r") as f:
    j_val.update(json.load(f))


def check_tags(tag_list: list) -> tuple[bool, typing.Union[Response, None]]:
    tag_schema = {
        {"type": "string"},
        {"type": "string"}
    }

    tag_list_schema = {
        "type": "array",
        "items": tag_schema
    }

    validator = Draft202012Validator(schema=tag_list_schema)

    v = validator.is_valid(tag_list)

    if not v:
        return False, Response(["Tag list does not follow proper structure"], status=400)
    else:
        return True, None


def check_type(v: typing.Any, class_str: str) -> tuple[bool, typing.Union[Response, None]]:
    def check_types(d, d_type):
        if type(d) != d_type:
            return False, Response(["Value: {} is not of type: {}".format(d, d_type)], status=400)
        else:
            return True, None

    match class_str:
        case "list":
            d_t = list
        case "int":
            d_t = int
        case "float":
            d_t = float
        case "str":
            d_t = str
        case "bool":
            d_t = bool
        case _:
            return False, Response(["Invalid Type provided: type({}) != {}".format(v, class_str)],
                                   status=400)

    return check_types(v, d_t)


def check_parameters(param_json: list) -> tuple[typing.Union[list, None], typing.Union[None, Response]]:
    param_schema = {
        "type": "object",
        "properties": {
            "param_name": {"type": "string"},
            "description": {"type": "string"},
            "default_value": {},
            "type": {"type": "string"},
            "options": {"type": ["null", "array"]},
            "is_forward": {"enum": [True, False]},
        },
        "required": ["param_name", "description", "default_value", "type", "options", "is_forward"]
    }

    schema = {
        "type": "array",
        "items": param_schema
    }

    validator = Draft202012Validator(schema=schema)

    v = validator.is_valid(param_json)

    if not v:
        x = [v0.message for v0 in sorted(validator.iter_errors(param_json), key=str)]
        return None, Response(x, status=400)

    for i in param_json:

        tp = i.get("type")
        val = i.get("default_value")

        j_field = {}

        if val:
            status, resp = check_type(val, tp)

            if status:
                j_field["default_value"] = val
            else:
                return None, resp

        val = i.get("options")

        val_list = []
        if val:
            for v in val:
                status, resp = check_type(v, tp)

                if status:
                    val_list.append(v)
                else:
                    return None, resp

            j_field["options"] = val_list

        i.pop("default_value")
        i.pop("options")

        i["default_and_options"] = j_field

    return param_json, None


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
    user = str(request.user)
    print(user)

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
@parser_classes([MultiPartParser, FormParser])
def insert_json_file(request):
    data = request.data

    data = {"upload": data["file"]}

    ser = JsonFileSerializer(data=data)

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
    name_set = set()
    has_forward_dict = data["forward_dict"]

    layer_data = {
        "layer_name": data.pop("layer_name"),
        "layer_file_id": data.pop("layer_file_id"),
        "layer_json_id": data.pop("layer_json_id"),
        "forward_list": data.pop("forward_list"),
        "forward_dict": has_forward_dict,
        "default": data.pop("default")
    }

    lyr_ser = LayerSerializer(data=layer_data)
    lyr_ser.is_valid(raise_exception=True)
    layer_id = lyr_ser.save().id

    out_list = data.pop("forward_output_names")
    f_list = [False] * len(out_list)

    if has_forward_dict:
        in_list = data.pop("forward_input_names")
        t_list = [True] * len(in_list)

        out_list += in_list
        f_list += t_list

    for d_dict, bol in zip(out_list, f_list):
        d_dict["layer_id"] = layer_id
        d_dict["is_output"] = bol

        name_len = len(name_set)

        name_set.add(d_dict["name"])

        if len(name_set) == name_len:
            return Response(["Duplicate name was used in input / output"], status=200)

    in_out_ser = InputOutputSerializer(data=out_list, many=True)

    if not in_out_ser.is_valid():
        instance = Layer.objects.get(id=layer_id)
        instance.delete()
        return Response(in_out_ser.errors, status=400)

    par_list = data.pop("parameter")

    for param_dict in par_list:
        param_dict["layer_id"] = layer_id
        name_len = len(name_set)
        name_set.add(param_dict["param_name"])
        if len(name_set) == name_len:
            return Response(["Duplicate name was used in input / output"], status=200)

    p_list, resp = check_parameters(par_list)

    if resp:
        return resp

    par_ser = LayerParameterSerializer(data=p_list,
                                       many=True,
                                       allow_null=True)

    if not par_ser.is_valid():
        instance = Layer.objects.get(id=layer_id)
        instance.delete()
        return Response(par_ser.errors, status=400)

    in_out_ser.save()
    par_ser.save()

    return Response(lyr_ser.data, status=200)


@api_view(["POST"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def insert_tag(request):
    user = str(request.user)

    model_data = get_user_model().objects.get(pk=user)
    if not model_data.is_staff:
        return Response({"invalid": "You are not a member of staff"}, status=400)

    request.data["tag_name"] = request.data["tag_name"].upper()
    request.data["tag_class_name"] = request.data["tag_class_name"].upper()
    serializer = TagSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):
        instance = serializer.save()
        tgc = instance.tag_class_name
        tgc.tag_count += 1
        tgc.save()
        return Response(serializer.data, status=200)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["POST"])
@permission_classes([HasAPIKey])
def insert_layer_tag(request):
    req_list = []
    for i in range(len(request.data)):
        request.data[i]["tag_name"] = request.data[i]["tag_name"].upper()

        item = Tag_Layer_Model.objects.filter(tag_name=request.data[i]["tag_name"],
                                              layer_id=request.data[i]["layer_id"])

        print(item)

        if not item.exists():
            req_list.append(request.data[i])

    serializer = TagLayerSerializer(data=req_list,
                                    many=True)

    if serializer.is_valid(raise_exception=True):

        tag_list = serializer.save()

        for tag in tag_list:
            tgn = tag.tag_name
            tgn.count += 1
            tgn.save()

        return Response(serializer.data, status=200)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)
