import copy
import math
import json
from django.core.exceptions import ValidationError
from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.postgres.fields import JSONField
from uuid import uuid5


# Create your models here.

def validate_output_json(output_json: dict):
    j_val = {}
    with open("../../../AdditionalFiles/fields.json") as f:
        j_val.update(json.load(f))

    def check_key(d_key: str) -> None:

        dtype_keys = set(j_val["valid_datatype"].keys())

        if d_key == "parameter_list":

            for k, v in output_json[d_key].items():
                if v["description"] != str:
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

            if d_key == "input_shape":

                for k, v in output_json[d_key].items():


    valid_keys = j_val["valid_keys"]
    pop_keys = copy.deepcopy(valid_keys)

    for key in output_json.keys():

        if key in pop_keys:
            pop_keys.pop(key)
        else:
            er_str = "Key {} is not a valid key, valid keys are: parameter_list, input_shape, output_shape"
            raise ValidationError(er_str)

    if len(valid_keys) != 0:
        er_str = "Key " if len(valid_keys) == 1 else "Key's "
        er_str += " {} are missing".format(str(valid_keys).strip("{}").replace("'", ""))
        raise ValidationError(er_str)


class LayerModel(models.Model):
    id = models.UUIDField(max_length=37,
                          default=uuid5,
                          editable=False,
                          primary_key=True)

    layer_name = models.CharField(max_length=100,
                                  help_text="the name of the layer")

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    upload = models.FileField(upload_to='uploads/layers',
                              validators=[validate_file_contents,
                                          FileExtensionValidator(['py'])])

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    layer_parameters = JSONField(validators=[validate_output_json])
