from random import random
from typing import Optional, TYPE_CHECKING
from Input import Input

if TYPE_CHECKING:
    from Layers import Layer

class Neuron:
    def __init__(self, layer: "Layer", prev_layer: Optional["Layer"] = None):
        self.layer: "Layer" = layer
        self.inputs: "list[Input] | list[float]" = (
            [Input(n, random() - 0.5) for n in prev_layer.neurons]
            if prev_layer else [0.0]
        )

    @property
    def isFirstLayerNeuron(self) -> bool:
        return not isinstance(self.inputs[0], Input)
    
    @property
    def inputSum(self) -> float:
        summ: float = 0.0
        for i in self.inputs:
            if isinstance(i, Input):
                summ += i.neuron.value * i.weight
            else:
                summ += 0
        return summ
    
    @property
    def value(self) -> float:
        if self.isFirstLayerNeuron:
            return self.inputs[0] # type: ignore
        return self.layer.network.activation_function(self.inputSum)
    
    def n_input(self, value: float) -> None:
        if not self.isFirstLayerNeuron:
            return
        self.inputs[0] = value # type: ignore

    def error(self, e: float) -> None:
        if self.isFirstLayerNeuron:
            return
        
        wDelta: float = e * self.layer.network.activation_function_derivative(self.inputSum)
        for i in self.inputs:
            if isinstance(i, Input):
                i.weight -= i.neuron.value * wDelta * self.layer.network.learning_rate
                i.neuron.error(i.weight * wDelta)