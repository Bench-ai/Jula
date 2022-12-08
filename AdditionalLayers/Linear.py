import typing
import torch
import torch.nn as nn
from AdditionalLayers.BaseLayer import InputLayer


class LinearLayer(InputLayer):

    def __init__(self,
                 target_shape: typing.Tuple[int, ...],
                 out_features: int,
                 bias=True):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        out_features: size of the output
        bias: Determines whether the layer will learn an additive bias.
        """

        self.__out_shape = out_features
        self.__tar_shape = target_shape

        super(LinearLayer, self).__init__(target_shape)
        self.__linear = nn.Linear(target_shape[-1],
                                  out_features,
                                  bias)

    def get_output_shape(self) -> int:
        """

        Returns the output shape of the layer
        -------

        """
        return self.__out_shape

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the sigmoid version of the layer
        -------

        """
        return self.__linear(super()(x))
