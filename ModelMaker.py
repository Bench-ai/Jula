import torch
from numpy.ma import copy
from ordered_set import OrderedSet
import json
from torch import nn
from LayerReader import LinearSwitch
from typing import Union
from AdditionalLayers import BaseLayer


def read_json(j_file: str) -> Union[list, dict]:
    """

    Parameters
    ----------
    j_file: Takes in a json file and returns a dictionary

    Returns
    -------

    """
    with open(j_file, "r") as file:
        layer_dict = json.load(file)

    return layer_dict


def get_seen_set(layer_dict: dict,
                 output_layer_id: list[str],
                 output_layer_dict: list[dict]):
    """

    Parameters
    ----------
    layer_dict: A dictionary detailing the layer format
    output_layer_id: A list of the output layer id's
    output_layer_dict: A list of the output layer dictionary's

    Returns
    -------

    """
    seen_set = OrderedSet()

    def iterate(in_layer_dict: dict,
                in_output_layer_id: str,
                in_output_layer_dict: dict) -> None:

        """

        Parameters
        ----------
        in_layer_dict: The layer_dict
        in_output_layer_id: the id of the layer
        in_output_layer_dict: the output layer dict

        Returns the order of traversal
        -------

        """

        if in_output_layer_dict["input_layer"] is not None:

            for i in in_output_layer_dict["input_layer"]:

                if i not in seen_set:
                    iterate(in_layer_dict,
                            str(i),
                            in_layer_dict[str(i)])

            seen_set.add(in_output_layer_id)

        else:
            seen_set.add(in_output_layer_id)

    for x, y in zip(output_layer_id, output_layer_dict):
        iterate(layer_dict,
                x,
                y)

    return seen_set


class MainModel(nn.Module):

    def __init__(self,
                 layer_json: dict,
                 seen_set: OrderedSet):
        """

        Parameters
        ----------
        layer_json: The dictionary of the layered json
        seen_set: the order of traversal of the neural network graph
        """

        super(MainModel, self).__init__()

        self.__seen_set = seen_set
        self.__module_dict = nn.ModuleDict()
        self.__layer_json = layer_json

        for l_id in self.__seen_set:
            layer_dict = copy.deepcopy(layer_json[l_id])
            layer_name = layer_dict["layer_data"].pop("name")

            self.__module_dict.update({
                l_id: LinearSwitch(layer_name, layer_dict["layer_data"]).get_layer()
            })

    def forward(self,
                input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        input_dict: A dictionary of the input id's corresponding to their input tensors

        Returns A dictionary of the output id's corresponding to the output tensors
        -------

        """

        layer_dictionary = {}
        output_dictionary = {}

        for l_id in self.__seen_set:

            if isinstance(self.__module_dict[l_id], BaseLayer.InputLayer):
                layer_dictionary[l_id] = self.__module_dict[l_id](input_dict[l_id])

            else:
                input_list = []
                for i in self.__layer_json[l_id]["input_layer"]:
                    input_list.append(layer_dictionary[i])

                layer_dictionary[l_id] = self.__module_dict[l_id](input_list)

            if not self.__layer_json[l_id]["output_layer"]:
                output_dictionary[l_id] = layer_dictionary[l_id]

        return output_dictionary


if __name__ == '__main__':
    my_layer_dict = read_json("FakeJson.json")

    my_output_layer_id_list = ["8"]

    my_output_layer_dict = [
                            my_layer_dict["8"]]

    print(get_seen_set(my_layer_dict,
                       my_output_layer_id_list,
                       my_output_layer_dict))
