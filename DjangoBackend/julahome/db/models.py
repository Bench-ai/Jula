import copy
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from django.core.exceptions import ValidationError
from django.db import models
from django.core.validators import FileExtensionValidator
from django.db.models import JSONField
from uuid import uuid4
import re
from django.conf import settings

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)


# Create your models here.


def py_content_validator(py_file):
    j_val = {}

    with open(os.getenv("FIELDS_PATH"), "r") as f:
        j_val.update(json.load(f))

    lib_set = set(j_val["illegal_libs"])

    lib_set.update(set(j_val["illegal_code"]))

    with open(py_file, "r") as file:
        line = file.readline()

        line_set = set(re.sub("[/(/).]", " ", line).split())

        total_len = len(line_set) + len(lib_set)

        line_set.update(lib_set)

        if len(line_set) != total_len:
            raise ValidationError("Illegal command is being used")


def validate_output_json(output_json: dict):
    j_val = {}

    with open(os.getenv("FIELDS_PATH"), "r") as f:
        j_val.update(json.load(f))

    dtype_keys = set(j_val["valid_datatype"].keys())
    allowed_keys = set(j_val["allowed_keywords"])
    un_allowed = j_val["not_allowed_chars"]
    all_ops = set(j_val["allowed_operations"])

    def check_p_list():

        for k, v in output_json["parameter_list"].items():

            if type(v["description"]) != str:
                raise ValidationError("{}'s, description must be a string.".format(k))

            if v["type"] not in dtype_keys:
                raise ValidationError("{}'s, type key {} is not a valid dtype.".format(k, v["type"]))

            if v["default"]:
                if type(v["default"]) is not eval(j_val["valid_datatype"][v["type"]]):
                    raise ValidationError(
                        "{}'s, type key {} and the default dtype don't match".format(k, v["type"]))

            if v["options"]:
                if type(v["options"]) is not list:
                    raise ValidationError("{}'s, options must be a list.".format(k))

    def check_shapes():

        cp_all_keys = copy.deepcopy(allowed_keys)
        cp_all_keys.update(set(output_json["parameter_list"].keys()))

        for idx, data_dict in enumerate(output_json["input_shape"]):

            if data_dict.get("name"):
                raise ValidationError("Input Shape {} has no name".format(idx))

            for shape_list in data_dict["shape"]:

                for var in shape_list[1]:

                    if var not in allowed_keys:
                        raise ValidationError("Variable {} of layer {} has no name".format(var,
                                                                                           data_dict.get("name")
                                                                                           ))
                if len(shape_list[0]) != len(re.sub(un_allowed, '', shape_list[0])):
                    raise ValidationError("Invalid character in layer {} formula".format(data_dict.get("name")))

                if len(shape_list) == 3 and shape_list[2]:
                    if shape_list[2] not in all_ops:
                        raise ValidationError("Invalid operation in layer {} formula".format(data_dict.get("name")))

    valid_keys = j_val["valid_keys"]
    pop_keys = copy.deepcopy(valid_keys)

    for key in output_json.keys():

        if key in pop_keys:
            key = pop_keys.pop(key)

            if key == "parameter_list":
                check_p_list()
            elif key == "output_shape" or key == "input_shape":
                check_shapes()
        else:
            er_str = "Key {} is not a valid key, valid keys are: parameter_list, input_shape, output_shape"
            raise ValidationError(er_str)

    if len(valid_keys) != 0:
        er_str = "Key " if len(valid_keys) == 1 else "Key's "
        er_str += " {} are missing".format(str(valid_keys).strip("{}").replace("'", ""))
        raise ValidationError(er_str)


User = settings.AUTH_USER_MODEL


class Layer(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    layer_name = models.CharField(max_length=100,
                                  help_text="the name of the layer")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    upload = models.FileField(upload_to='uploads/layers',
                              validators=[py_content_validator,
                                          FileExtensionValidator(['py'])])

    # user_name = models.ForeignKey(User, on_delete=models.PROTECT, default="7dIBzIUoMo")

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    layer_parameters = JSONField(validators=[validate_output_json])

    download_count = models.BigIntegerField(default=0)


class Tag_Class(models.Model):
    tag_class_name = models.CharField(max_length=100,
                                      help_text="the name of the tag",
                                      unique=True,
                                      blank=False,
                                      primary_key=True)

    user_name = models.ForeignKey(User, on_delete=models.PROTECT, default="7dIBzIUoMo")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    tag_count = models.BigIntegerField(default=0,
                                       help_text="How many tags belong to this specific class")


class Tag(models.Model):
    # id = models.UUIDField(max_length=37,
    #                       default=uuid4,
    #                       editable=False,
    #                       primary_key=True)

    tag_name = models.CharField(max_length=100,
                                help_text="the name of the tag",
                                primary_key=True)

    tag_class_name = models.ForeignKey(Tag_Class,
                                       on_delete=models.CASCADE,
                                       default="Z72tqCGEk4")

    # user_name = models.ForeignKey(User, on_delete=models.PROTECT, default="7dIBzIUoMo")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    count = models.BigIntegerField(default=0,
                                   help_text="How many class instances use this tag")


class TagLayerModel:
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    tag_name = models.ForeignKey(Tag,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    layer_id = models.ForeignKey(Layer,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)
