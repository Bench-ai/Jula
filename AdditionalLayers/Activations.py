import typing
import torch
import torch.nn as nn
from AdditionalLayers.BaseLayer import InputLayer


class SigmoidLayer(InputLayer):

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int]):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        """
        super(SigmoidLayer, self).__init__(target_shape)
        self.__sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the sigmoid version of the layer
        -------

        """
        return self.__sigmoid(super()(x))


class ReluLayer(InputLayer):

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int],
                 inplace: bool):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        inplace: Indicates whether you want the relu function to be inplace or not
        """
        super(ReluLayer, self).__init__(target_shape)
        self.__relu = nn.ReLU(inplace=inplace)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the Rectified Linear Unit
        -------

        """
        return self.__relu(super()(x))


class TanhLayer(InputLayer):

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int]):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        """
        super(TanhLayer, self).__init__(target_shape)
        self.__tanh = nn.Tanh()

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the hyperbolic tangent function
        -------

        """
        return self.__tanh(super()(x))


class SoftMaxLayer(InputLayer):

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int],
                 dim: typing.Optional[int]):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        dim: the dimension on which the softmax will be computed
        """
        super(SoftMaxLayer, self).__init__(target_shape)
        self.__softmax = nn.Softmax(dim)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the hyperbolic tangent function
        -------

        """
        return self.__softmax(super()(x))
