import typing

from AdditionalLayers.BaseLayer import InputLayer
from AdditionalLayers.Activations import SigmoidLayer, SoftMaxLayer, ReluLayer, TanhLayer
from AdditionalLayers.FunctionalLayers import ConcatenationLayer
from AdditionalLayers.Linear import LinearLayer


class LayerSwitch:

    def __init__(self):

        self.__parameter_dict = None

    def get_layer(self,
                  layer_name: str,
                  parameter_dict: dict) -> typing.Union[InputLayer,
                                                        SoftMaxLayer,
                                                        SigmoidLayer,
                                                        ReluLayer,
                                                        TanhLayer,
                                                        ConcatenationLayer,
                                                        LinearLayer]:
        """
        Parameters
        ----------
        layer_name: The name of the layer you wish to receive
        parameter_dict: The contents of that layers parameters

        Returns
        -------

        """

        self.__parameter_dict = parameter_dict
        return getattr(self, "{}".format(layer_name))()

    def InputLayer(self) -> InputLayer:
        """
        Returns the InputLayer
        -------
        """
        return InputLayer(**self.__parameter_dict)

    def SigmoidLayer(self) -> SigmoidLayer:
        """
        Returns the SigmoidLayer
        -------
        """
        return SigmoidLayer(**self.__parameter_dict)

    def SoftMaxLayer(self) -> SoftMaxLayer:
        """
        Returns the SoftmaxLayer
        -------

        """
        return SoftMaxLayer(**self.__parameter_dict)

    def ReluLayer(self) -> ReluLayer:
        """
        Returns the Relu Layer
        -------

        """
        return ReluLayer(**self.__parameter_dict)

    def TanhLayer(self) -> TanhLayer:
        """
        Returns the Tanh layer
        -------

        """
        return TanhLayer(**self.__parameter_dict)

    def ConcatenationLayer(self) -> ConcatenationLayer:
        """
        Returns the Concatenation Layer
        -------

        """
        return ConcatenationLayer(**self.__parameter_dict)

    def LinearLayer(self) -> LinearLayer:
        """
        Returns the Linear Layer
        -------

        """
        return LinearLayer(**self.__parameter_dict)
