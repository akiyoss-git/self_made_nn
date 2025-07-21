from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Neuron import Neuron
    from Network import Network

class Layer:
    def __init__(self, network: "Network", neuronsCount: int, prevLayer: Optional["Layer"] = None):
        from Neuron import Neuron  # локальный импорт, чтобы избежать цикла
        self.network: "Network" = network
        self.neurons: list["Neuron"] = [Neuron(self, prevLayer) for _ in range(neuronsCount)]
    
    @property
    def isFirstLayer(self) -> bool:
        return self.neurons[0].isFirstLayerNeuron
    
    def l_input(self, value: list[float]) -> None:
        if not self.isFirstLayer:
            return
        if len(value) != len(self.neurons):
            raise ValueError("Input length must match the number of neurons in the layer.")
        for i in range(len(value)):
            self.neurons[i].n_input(value[i])

