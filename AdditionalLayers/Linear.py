import torch
import torch.nn as nn


class LinearLayer(nn.Module):

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

        super(LinearLayer, self).__init__()

        self.__linear = nn.Linear(in_features,
                                  out_features,
                                  bias)

    def forward(self,
                x: list[torch.Tensor]) -> torch.Tensor:

        x = x[0]
        """

        Parameters
        ----------
        x: a list of tensors being fed into the model

        Returns the fully connected output of the layer
        -------

        """
        return self.__linear(x)
