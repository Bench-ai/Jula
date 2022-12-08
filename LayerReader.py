from AdditionalLayers.BaseLayer import InputLayer
from AdditionalLayers.Activations import SigmoidLayer, SoftMaxLayer, ReluLayer, TanhLayer
from AdditionalLayers.FunctionalLayers import ConcatenationLayer
from AdditionalLayers.Linear import LinearLayer


class LinearSwitch:

    def __init__(self,
                 layer_name: str,
                 parameter_dict: dict):

        """

        Parameters
        ----------
        layer_name: the name of the layer which the user wishes to receive
        parameter_dict: the parameters required for the layer
        """

        self.__layer_name = layer_name
        self.__parameter_dict = parameter_dict

    def get_layer(self):
        """
        Returns: The layer the user chose in the initialization method
        -------
        """
        return getattr(self, "__{}".format(self.__layer_name))

    def __InputLayer(self):
        """
        Returns the InputLayer
        -------
        """
        return InputLayer(**self.__parameter_dict)

    def __SigmoidLayer(self):
        """
        Returns the SigmoidLayer
        -------
        """
        return SigmoidLayer(**self.__parameter_dict)

    def __SoftMaxLayer(self):
        """
        Returns the SoftmaxLayer
        -------

        """
        return SoftMaxLayer(**self.__parameter_dict)

    def __ReluLayer(self):
        """
        Returns the Relu Layer
        -------

        """
        return ReluLayer(**self.__parameter_dict)

    def __TanhLayer(self):
        """
        Returns the Tanh layer
        -------

        """
        return TanhLayer(**self.__parameter_dict)

    def __ConcatenationLayer(self):
        """
        Returns the Concatenation Layer
        -------

        """
        return ConcatenationLayer(**self.__parameter_dict)

    def __LinearLayer(self):
        """
        Returns the Linear Layer
        -------

        """
        return LinearLayer(**self.__parameter_dict)
