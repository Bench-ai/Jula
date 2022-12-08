import typing


class IncorrectShapeError(Exception):
    # Exception raised for errors in the input shape

    def __init__(self,
                 target_shape: typing.Tuple[int, ...],
                 shape: typing.Tuple[int, ...]):
        """
        Parameters
        ----------
        target_shape: The desrired shape of the tensor
        shape: the actual shape of the tensor
        """
        t_str = ("Input shape (Any, " + ("{}, " * len(target_shape))[:-2] + ")").format(*shape)
        a_str = ("does not match target shape " + "(Any, " + ("{}, " * len(shape))[:-2] + ")").format(*target_shape)

        self.__message = t_str + " " + a_str

        super().__init__(self.__message)

