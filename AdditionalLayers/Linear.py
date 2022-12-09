import torch
import torch.nn as nn
from AdditionalLayers.BaseLayer import check_size


class LinearLayer(nn.Module):

    def __init__(self,
                 target_shape: int,
                 out_features: int,
                 bias=True):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        out_features: size of the output
        bias: Determines whether the layer will learn an additive bias.
        """

        super(LinearLayer, self).__init__()
        self.__out_shape = out_features
        self.__tar_shape = torch.Size([target_shape])

        self.__linear = nn.Linear(target_shape,
                                  self.__out_shape,
                                  bias)

    def get_output_shape(self) -> int:
        """

        Returns the output shape of the layer
        -------

        """
        return self.__out_shape

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
        check_size(x.size()[1:], self.__tar_shape)
        return self.__linear(x)
