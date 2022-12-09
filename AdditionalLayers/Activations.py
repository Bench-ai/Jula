import torch
import torch.nn as nn
from AdditionalLayers.BaseLayer import check_size


class SigmoidLayer(nn.Module):

    def __init__(self,
                 target_shape: tuple[int, ...]):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        """
        super(SigmoidLayer, self).__init__()
        self.__sigmoid = nn.Sigmoid()
        self.__target_shape = torch.Size(target_shape)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the sigmoid function
        -------

        """
        check_size(x.size()[1:], self.__target_shape)
        return self.__sigmoid(x)


class ReluLayer(nn.Module):

    def __init__(self,
                 target_shape: tuple[int, ...],
                 inplace=False):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        inplace: Indicates whether you want the relu function to be inplace or not
        """
        super(ReluLayer, self).__init__()
        self.__relu = nn.ReLU(inplace=inplace)
        self.__target_shape = torch.Size(target_shape)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the Rectified Linear Unit
        -------

        """
        check_size(x.size()[1:], self.__target_shape)
        return self.__relu(x)


class TanhLayer():

    def __init__(self,
                 target_shape: tuple[int, ...]):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        """
        super(TanhLayer, self).__init__()
        self.__tanh = nn.Tanh()
        self.__target_shape = torch.Size(target_shape)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the hyperbolic tangent function
        -------

        """
        check_size(x.size()[1:], self.__target_shape)
        return self.__tanh(x)


class SoftMaxLayer(nn.Module):

    def __init__(self,
                 target_shape: tuple[int, ...],
                 dim: int):
        """
        Parameters
        ----------
        target_shape: the Expected shape of the input tensor
        dim: the dimension on which the softmax will be computed
        """
        super(SoftMaxLayer, self).__init__()
        self.__softmax = nn.Softmax(dim)
        self.__target_shape = torch.Size(target_shape)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: the tensor being fed into the layer

        Returns the layer activated by the Softmax function
        -------

        """
        check_size(x.size()[1:], self.__target_shape)
        return self.__softmax(super()(x))
