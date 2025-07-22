from Layers import Layer
from math import ceil
from typing import Callable
from ActivationFunctions import ActivationFunctions

class Network:
    def __init__(self, input_size: int, output_size: int, hidden_layers_count: int = 1, learning_rate: float = 0.5, activation_function: str = "linear"):
        functions = ActivationFunctions.get_functions(activation_function)
        self.activation_function: Callable[[float], float] = functions[0]
        self.activation_function_derivative: Callable[[float], float] = functions[1]
        self.learning_rate: float = learning_rate

        self.layers = [Layer(self, input_size, None)]
        for _ in range(hidden_layers_count):
            layer_size = min(input_size * 2 - 1, ceil((input_size * 2 / 3) + output_size))
            self.layers.append(Layer(self, layer_size, self.layers[-1]))
        self.layers.append(Layer(self, output_size, self.layers[-1]))

    def nn_input(self, value: list[float]) -> None:
        self.layers[0].l_input(value)
    
    @property
    def prediction(self):
        return [n.value for n in self.layers[-1].neurons]
    
    def train_once(self, dataset: list[list[list[float]]]) -> None:
        for case in dataset:
            input_data, expected_output = case
            self.nn_input(input_data)
            for i in range(len(self.prediction)):
                self.layers[-1].neurons[i].error(self.prediction[i] - expected_output[i])

    def train(self, dataset: list[list[list[float]]], epochs: int = 1000):
        for _ in range(epochs):
            self.train_once(dataset)

data: list[list[list[float]]] = [
 [[0, 0], [0]],
 [[0, 1], [1]],
 [[1, 0], [1]],
 [[1, 1], [0]],
]
testData: list[list[float]] = [
   [0, 0],
   [0, 1],
   [1, 0],
   [1, 1],
 ]

logistic = Network(2, 1, 0, activation_function="sigmoid")

logistic.train(data, epochs=10000)

for test in testData:
    logistic.nn_input(test)
    print(f"SIGMOID : LOGISTIC :: Input: {test}, Prediction: {logistic.prediction}")

MLP_1 = Network(2, 1, 1, activation_function="sigmoid")

MLP_1.train(data, epochs=10000)

for test in testData:
    MLP_1.nn_input(test)
    print(f"SIGMOID : MLP :: Input: {test}, Prediction: {MLP_1.prediction}")

MLP_2 = Network(2, 1, 2, activation_function="sigmoid")
MLP_2.train(data, epochs=10000)
for test in testData:
    MLP_2.nn_input(test)
    print(f"SIGMOID : MLP 2 :: Input: {test}, Prediction: {MLP_2.prediction}")

