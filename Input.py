from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Neuron import Neuron

class Input:
    def __init__(self, neuron: "Neuron", weight: float):
        self.neuron: "Neuron" = neuron
        self.weight: float = weight