import torch
import torch.nn as nn


class TestLinearLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True):
        """
        Parameters
        ----------
        in_features: size of the input features
        out_features: size of the output features
        bias: Determines whether the layer will learn an additive bias.
        """

        super(TestLinearLayer, self).__init__()

        self.__linear = nn.Linear(in_features,
                                  out_features,
                                  bias)

    def forward(self,
                input_dict: dict[str, torch.Tensor],
                variable_dict: dict) -> dict[str, torch.Tensor]:

        print(variable_dict["test_bool"])

        x = input_dict["input"]
        """

        Parameters
        ----------
        x: a list of tensors being fed into the model

        Returns the fully connected output of the layer
        -------

        """
        return {"output": self.__linear(x)}
