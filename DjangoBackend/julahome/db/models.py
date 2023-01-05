import os
from django.utils.deconstruct import deconstructible
from dotenv import load_dotenv
from pathlib import Path
import json
from django.core.exceptions import ValidationError
from django.db import models
from django.core.validators import FileExtensionValidator
from uuid import uuid4
import re
from django.conf import settings
from jsonschema import Draft202012Validator

dotenv_path = Path('D:/BenchAi/Jula/.env')
load_dotenv(dotenv_path=dotenv_path)

j_val = {}
with open(os.getenv("FIELDS_PATH"), "r") as f:
    j_val.update(json.load(f))


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


# Create your models here.

@deconstructible
class PyValidator:
    __lib_set = set(j_val["illegal_libs"])
    __lib_set.update(set(j_val["illegal_code"]))
    __max_size = 5_000_000

    def __call__(self, data):

        if self.__max_size < data.size:
            raise ValidationError("file size {} is greater than Maximum allowed file size {}".format(data.size,
                                                                                                     self.__max_size))

        for line in data:

            line = line.decode('utf-8')

            line_set = set(re.sub("[/(/).]", " ", line).split())

            total_len = len(line_set) + len(self.__lib_set)

            line_set.update(self.__lib_set)

            if len(line_set) != total_len:
                raise ValidationError("Illegal command is being used")

    def __eq__(self, other):
        return True


@deconstructible
class JsonValidator:
    __max_size = 5_000_000

    __var_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "parameter": {"type": "boolean"},
            "type": {"enum": ["list", "int", "str", "bool", "equation"]},
            "value": {"type": ["string", "null"]}
        },
        "required": ["name", "parameter", "type", "value"],
        "additionalProperties": False
    }

    __var_pre_1 = {
        "type": "object",
        "properties": {
            "name": {"const": "ret_bool"},
            "parameter": {"const": False},
            "type": {"const": "bool"},
            "value": {"enum": ["False", "True"]}
        },
        "required": ["name", "parameter", "type", "value"],
        "additionalProperties": False
    }

    __var_pre_2 = {
        "type": "object",
        "properties": {
            "name": {"const": "ret_str"},
            "parameter": {"const": False},
            "type": {"const": "str"},
            "value": {"const": "valid"}
        },
        "required": ["name", "parameter", "type", "value"],
        "additionalProperties": False
    }

    __variables_schema = {
        "type": "array",
        "minItems": 2,
        "prefixItems": [
            __var_pre_1,
            __var_pre_2
        ],
        "items": __var_item_schema
    }

    __equation_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "equation_str": {"type": "string"},
            "variables": {"type": "array",
                          "items": {"type": "string"}},
        },
        "required": ["name", "equation_str", "variables"],
        "additionalProperties": False
    }

    __equation_schema = {
        "type": "array",
        "items": __equation_item_schema
    }

    __command_string_schema = {
        "type": "array",
        "items": {"type": "string"}
    }

    __return_var_schema = {
        "type": "array",
        "prefixItems": [
            {"const": "ret_bool"},
            {"const": "ret_str"}
        ]
    }

    __block_item = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "code": {"type": "string"},
            "is_var": {"type": "boolean"},
        },
        "required": ["name", "code", "is_var"],
        "additionalProperties": False
    }

    __block_schema = {
        "type": "array",
        "items": __block_item
    }

    __input_schema = {
        "type": "object",
        "properties": {
            "variables": __variables_schema,
            "equations": __equation_schema,
            "command_strings": __command_string_schema,
            "blocks": __block_schema,
            "return_vars": __return_var_schema
        },
    }

    __return_var_schema_2 = {
        "type": "array",
        "minItems": 3,
        "prefixItems": [
            {"const": "ret_bool"},
            {"const": "ret_str"}
        ],
        "items": {
            "type": "string"
        }
    }

    __input_schema_2 = {
        "type": "object",
        "properties": {
            "variables": __variables_schema,
            "equations": __equation_schema,
            "command_strings": __command_string_schema,
            "blocks": __block_schema,
            "return_vars": __return_var_schema_2
        },
    }

    @staticmethod
    def __check_script(script: dict,
                       validator) -> None:

        v = validator.is_valid(script)

        if not v:
            x = [v0.message for v0 in sorted(validator.iter_errors(script), key=str)]
            raise ValidationError(x)

    def __call__(self, data):
        if self.__max_size < data.size:
            raise ValidationError("file size {} is greater than Maximum allowed file size {}".format(data.size,
                                                                                                     self.__max_size))
        try:
            json_dict = json.load(data)
        except json.decoder.JSONDecodeError as e:
            raise ValidationError(e)

        validator = Draft202012Validator(schema=self.__input_schema)

        has_input_script = False

        script = json_dict.get("input_script")

        if script:
            has_input_script = True
            self.__check_script(script, validator)

        script = json_dict.get("output_script")

        validator = validator.evolve(schema=self.__input_schema_2)

        if script:
            self.__check_script(script, validator)

        elif not has_input_script:
            raise ValidationError("Neither a input nor output script has been provided")

    def __eq__(self, other):
        return True


User = settings.AUTH_USER_MODEL


class Layer_File(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    p_val = PyValidator()

    upload = models.FileField(upload_to='uploads/layers',
                              validators=[p_val,
                                          FileExtensionValidator(['py'])])

    update_timestamp = models.DateTimeField(auto_now=True)

    download_count = models.BigIntegerField(default=0)


class Json_File(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    j_val = JsonValidator()

    upload = models.FileField(upload_to='uploads/Json',
                              validators=[j_val,
                                          FileExtensionValidator(['json'])])

    update_timestamp = models.DateTimeField(auto_now=True)


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

    layer_json_id = models.ForeignKey(Json_File,
                                      on_delete=models.PROTECT,
                                      default=uuid4)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    default = models.BooleanField(default=False, null=False, blank=False)

    forward_list = models.BooleanField(null=True, blank=False)

    forward_dict = models.BooleanField(null=False, blank=False, default=False)


class Layer_Parameter(models.Model):
    DATA_TYPE_CHOICES = [
        ('int', 'integer'),
        ('bool', 'boolean'),
        ('float', 'float'),
        ('str', 'string'),
        ('list', "array"),
        ("dict", "Map")
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

    type = models.CharField(max_length=5,
                            choices=DATA_TYPE_CHOICES,
                            blank=False)

    default_and_options = models.JSONField(blank=True,
                                           help_text="This will be a list of values all of self.type",
                                           null=True)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    is_forward = models.BooleanField(default=False, null=False, blank=False)


class input_output_details(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid4,
                          editable=False,
                          primary_key=True)

    layer_id = models.ForeignKey(Layer,
                                 on_delete=models.PROTECT,
                                 default=uuid4)

    name = models.CharField(max_length=100,
                            blank=False)

    description = models.TextField(help_text="the name of the tag",
                                   blank=False)

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    is_output = models.BooleanField(null=False, blank=False)


# not used
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


# not used
class Input_Output_Channels(models.Model):
    OPERATION_CHOICES = [
        ("CEIL", "ceil"),
        ("FLOR", "floor"),
        ("AANY", "any"),
        ("COPY", "copy")
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
                                 blank=True,
                                 null=True)

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
