import os
from dotenv import load_dotenv
from pathlib import Path
import json
from django.core.exceptions import ValidationError
from django.db import models
from django.core.validators import FileExtensionValidator
from uuid import uuid4
import re
from django.conf import settings

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)

j_val = {}
with open(os.getenv("FIELDS_PATH"), "r") as f:
    j_val.update(json.load(f))


# Create your models here.


def py_content_validator(py_file):
    lib_set = set(j_val["illegal_libs"])

    lib_set.update(set(j_val["illegal_code"]))

    for line in py_file:

        line = line.decode('utf-8')

        line_set = set(re.sub("[/(/).]", " ", line).split())

        total_len = len(line_set) + len(lib_set)

        line_set.update(lib_set)

        if len(line_set) != total_len:
            raise ValidationError("Illegal command is being used")



def validate_p_list(output_json: dict):
    dtype_keys = set(j_val["valid_datatype"].keys())

    for k, v in output_json.items():

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


def check_shapes(output_json: list, parameter_keys: list):
    cp_all_keys = set(j_val["allowed_keywords"])
    cp_all_keys.update(set(parameter_keys))

    un_allowed = j_val["not_allowed_chars"]
    all_ops = set(j_val["allowed_operations"])

    for idx, data_dict in enumerate(output_json):

        if data_dict.get("name"):
            raise ValidationError("Input Shape {} has no name".format(idx))

        for shape_list in data_dict["shape"]:
            for var in shape_list[1]:

                if var not in cp_all_keys:
                    raise ValidationError("Variable {} of layer {} is not valid".format(var,
                                                                                        data_dict.get("name")
                                                                                        ))
                if len(shape_list[0]) != len(re.sub(un_allowed, '', shape_list[0])):
                    raise ValidationError("Invalid character in layer {} formula".format(data_dict.get("name")))

                if len(shape_list) == 3 and shape_list[2]:
                    if shape_list[2] not in all_ops:
                        raise ValidationError("Invalid operation in layer {} formula".format(data_dict.get("name")))


User = settings.AUTH_USER_MODEL


class Layer_File(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    upload = models.FileField(upload_to='uploads/layers',
                              validators=[py_content_validator,
                                          FileExtensionValidator(['py'])])

    update_timestamp = models.DateTimeField(auto_now=True)

    download_count = models.BigIntegerField(default=0)


class Layer(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    layer_name = models.CharField(max_length=100,
                                  help_text="the name of the layer")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    layer_file_id = models.ForeignKey(Layer_File,
                                      on_delete=models.PROTECT,
                                      default=uuid4)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)


class Layer_Parameter(models.Model):
    DATA_TYPE_CHOICES = [
        ('INT', 'integer'),
        ('BOL', 'boolean'),
        ('FLO', 'float'),
        ('STR', 'string'),
    ]

    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    layer_id = models.ForeignKey(Layer,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    param_name = models.CharField(max_length=100,
                                  blank=False)

    description = models.TextField(help_text="the name of the tag",
                                   blank=False)

    default_value = models.CharField(max_length=100,
                                     blank=True)

    type = models.CharField(max_length=3,
                            choices=DATA_TYPE_CHOICES,
                            blank=False)

    options = models.JSONField(blank=True,
                               help_text="This will be a list of values all of self.type")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)


class Layer_Input_Output(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    layer_id = models.ForeignKey(Layer,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    input_name = models.CharField(max_length=100,
                                  blank=False)

    is_input = models.BooleanField(blank=False)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)


class Input_Output_Channels(models.Model):

    OPERATION_CHOICES = [
        ("CEIL", "ceil"),
        ("FLOR", "floor")
    ]

    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    input_id = models.ForeignKey(Layer_Input_Output,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    input_shape_str = models.CharField(max_length=200,
                                       blank=False)

    variables = models.JSONField(blank=False,
                                 help_text="This will be a list of values to fill into the input_shape_str")

    channel_number = models.IntegerField(blank=False)

    operation = models.CharField(max_length=4,
                                 choices=OPERATION_CHOICES,
                                 blank=True)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)


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

    tag_name = models.CharField(max_length=100,
                                help_text="the name of the tag",
                                primary_key=True)

    tag_class_name = models.ForeignKey(Tag_Class,
                                       on_delete=models.CASCADE,
                                       default="Z72tqCGEk4")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    count = models.BigIntegerField(default=0,
                                   help_text="How many class instances use this tag")


class Tag_Layer_Model(models.Model):
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
