import typing
import AdditionalLayers.Exceptions as e
import torch
import torch.nn as nn


class InputLayer(nn.Module):

    def __init__(self,
                 target_shape: typing.Tuple[int, ...]):
        """
        Parameters
        ----------
        target_shape: the shape of the input tensor
        """

        super(InputLayer, self).__init__()
        self.__target_shape = target_shape

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: Tensor being passed into the model

        Returns the exact same Tensor
        -------

        """
        if tuple(x.size()[1:]) != self.__target_shape:
            raise e.IncorrectShapeError(self.__target_shape,
                                        x.size()[1:])

        return x

    def get_output_shape(self) -> typing.Tuple[int, ...]:
        """
        Returns the target_size since no changes to the size occur
        -------
        """

        return self.__target_shape


class InputLayer2(nn.Module):

    def __init__(self,
                 target_shape_list: typing.List[typing.Tuple[int, ...]]):
        """
        Parameters
        ----------
        target_shape_list: A list of all tensor shapes that will be passed in.
        """

        super(InputLayer2, self).__init__()
        self.__target_shape_list = target_shape_list

    def forward(self,
                tensor_list: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        """

        Parameters
        ----------
        tensor_list: A list of tensors that will be passed into the model

        Returns the exact same Tensor List
        -------

        """

        for idx, ten in enumerate(tensor_list):

            if tuple(ten.size()[1:]) != self.__target_shape_list[idx]:
                raise e.IncorrectShapeError(self.__target_shape_list[idx],
                                            ten.size()[1:])

        return tensor_list

    def get_output_shape(self) -> typing.Tuple[int, ...]:
        """
        Returns the size of the first target tensor since no changes occur
        -------
        """

        return self.__target_shape_list[0]
