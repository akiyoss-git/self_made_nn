from math import exp
from typing import Callable

class ActivationFunctions:
    @staticmethod
    def linear(x: float) -> float:
        return x
    
    @staticmethod
    def linear_derivative(x: float) -> float:
        return 1.0

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        sig = ActivationFunctions.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(x: float) -> float:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        return 1 - ActivationFunctions.tanh(x) ** 2
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def leaky_relu(x: float) -> float:
        return max(0.01 * x, x)
    
    @staticmethod
    def leaky_relu_derivative(x: float) -> float:
        return 0.01 if x < 0 else 1.0
    
    @staticmethod
    def parametric_relu(x: float) -> float:
        return max(0.5 * x, x)
    
    @staticmethod
    def parametric_relu_derivative(x: float) -> float:
        return 0.5 if x < 0 else 1.0
    
    @staticmethod
    def ELU(x: float) -> float:
        return x if x >= 0 else 0.1 * (exp(x) - 1)
    
    @staticmethod
    def ELU_derivative(x: float) -> float:
        return 1 if x >= 0 else 0.1 * exp(x)
    
    @staticmethod
    def get_functions(name: str) -> list[Callable[[float], float]]:
        functions = {
            "linear": [ActivationFunctions.linear, ActivationFunctions.linear_derivative],
            "sigmoid": [ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative],
            "tanh": [ActivationFunctions.tanh, ActivationFunctions.tanh_derivative],
            "relu": [ActivationFunctions.relu, ActivationFunctions.relu_derivative],
            "leaky_relu": [ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative],
            "parametric_relu": [ActivationFunctions.parametric_relu, ActivationFunctions.parametric_relu_derivative],
            "ELU": [ActivationFunctions.ELU, ActivationFunctions.ELU_derivative],
        }
        return functions.get(name, [ActivationFunctions.linear, ActivationFunctions.linear_derivative])
    