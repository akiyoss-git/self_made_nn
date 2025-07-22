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

class ConvolutionalLayer(Layer):
    def __init__(
        self,
        network: "Network",
        input_size: int,
        kernel_size: int = 3,
        num_kernels: int = 1,
        stride: int = 1,
        padding: int = 0,
        prevLayer: Optional["Layer"] = None
    ):
        super().__init__(network, input_size, prevLayer)
        import random
        self.kernel_size: int = kernel_size
        self.num_kernels: int = num_kernels
        self.stride: int = stride
        self.padding: int = padding
        self.kernels: list[list[float]] = [
            [random.uniform(-0.5, 0.5) for _ in range(kernel_size)]
            for _ in range(num_kernels)
        ]
        self.biases: list[float] = [random.uniform(-0.5, 0.5) for _ in range(num_kernels)]
        self.output_size: int = ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, input_data: list[float]) -> list[float] | list[list[float]]:
        padded = [0.0] * self.padding + input_data + [0.0] * self.padding
        outputs: list[list[float]] = []
        for k in range(self.num_kernels):
            kernel = self.kernels[k]
            bias = self.biases[k]
            out: list[float] = []
            for i in range(0, len(padded) - self.kernel_size + 1, self.stride):
                window = padded[i:i+self.kernel_size]
                conv = sum(w * x for w, x in zip(kernel, window)) + bias
                # Пример: ReLU-активация
                conv = max(0, conv)
                out.append(conv)
            outputs.append(out)
        return outputs[0] if self.num_kernels == 1 else outputs


class MaxPooling(Layer):
    def __init__(self, network: "Network", input_size: int, pool_size: int = 2, stride: int = 2, prevLayer: Optional["Layer"] = None):
        super().__init__(network, input_size, prevLayer)
        self.pool_size: int = pool_size
        self.stride: int = stride
        self.output_size: int = (input_size - pool_size) // stride + 1

    def forward(self, input_data: list[float]) -> list[float]:
        outputs: list[float] = []
        for i in range(0, len(input_data) - self.pool_size + 1, self.stride):
            window = input_data[i:i+self.pool_size]
            outputs.append(max(window))
        return outputs
    
class Flatten(Layer):
    def __init__(self, network: "Network", input_size: int, prevLayer: Optional["Layer"] = None):
        super().__init__(network, input_size, prevLayer)

    def forward(self, input_data: list[list[float]]) -> list[float]:
        return [item for sublist in input_data for item in sublist]
    
