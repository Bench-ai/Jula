import AdditionalLayers.Exceptions as e
import torch
import torch.nn as nn


def check_size(x_size: torch.Size,
               y_size: torch.Size):
    if x_size != y_size:
        raise e.IncorrectShapeError(x_size,
                                    y_size)


def check_multi_size(x_size_list: list[torch.Size],
                     y_size_list: list[torch.Size]):
    for i, j in zip(x_size_list, y_size_list):
        check_size(i, j)


class InputLayer(nn.Module):

    def __init__(self,
                 target_shape: tuple[int, ...]):
        """
        Parameters
        ----------
        target_shape: the shape of the input tensor
        """

        super(InputLayer, self).__init__()
        self.__target_shape = torch.Size(target_shape)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: Tensor being passed into the model

        Returns the exact same Tensor
        -------

        """

        check_size(x.size()[1:], self.__target_shape)

        return x

    def get_output_shape(self) -> tuple[int, ...]:
        """
        Returns the target_size since no changes to the size occur
        -------
        """

        return self.__target_shape
