from copy import deepcopy
import torch
import torch.nn as nn
from AdditionalLayers.BaseLayer import check_multi_size


class ConcatenationLayer(nn.Module):

    def __init__(self,
                 dim: int):
        """
        Parameters
        ----------
        target_shape_list: A list of all tensor shapes that will be passed in.
        dim: the dimension the concatenation occurs
        """

        super(ConcatenationLayer, self).__init__()
        # self.__target_shape_list = [torch.Size(i) for i in target_shape_list]
        self.__dim = dim

    def forward(self,
                tensor_list: list[torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        tensor_list: The list of Tensors that you wish to concatenate

        Returns the exact same Tensor
        -------

        """

        x_size_list = [x.size()[1:] for x in tensor_list]
        check_multi_size(x_size_list, self.__target_shape_list)

        return torch.cat(tensor_list, dim=self.__dim)

    def get_output_shape(self) -> tuple[int, ...]:
        """
        Returns the new output size of the Tensor after concatenating based on the dimension
        -------
        """

        output_shape = list(deepcopy(self.__target_shape_list[0]))

        for size in self.__target_shape_list[1:]:
            output_shape[self.__dim] += list(size[self.__dim])

        return tuple(output_shape)
