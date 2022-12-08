import typing
from copy import deepcopy
import torch
from AdditionalLayers.BaseLayer import InputLayer2


class ConcatenationLayer(InputLayer2):

    def __init__(self,
                 target_shape_list: typing.List[typing.Tuple[int, ...]],
                 dim: int):
        """
        Parameters
        ----------
        target_shape_list: A list of all tensor shapes that will be passed in.
        dim: the dimension the concatenation occurs
        """

        super(ConcatenationLayer, self).__init__(target_shape_list)
        self.__target_shape_list = target_shape_list
        self.__dim = dim

    def forward(self,
                tensor_list: typing.List[torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        tensor_list: The list of Tensors that you wish to concatenate

        Returns the exact same Tensor
        -------

        """

        tensor_list = super(tensor_list)

        return torch.cat(tensor_list, dim=self.__dim)

    def get_output_shape(self) -> typing.Tuple[int, ...]:
        """
        Returns the new output size of the Tensor after concatenating based on the dimension
        -------
        """

        output_shape = list(deepcopy(self.__target_shape_list[0]))

        for size in self.__target_shape_list[1:]:
            output_shape[self.__dim] += size[self.__dim]

        return tuple(output_shape)
