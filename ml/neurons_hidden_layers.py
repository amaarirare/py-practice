import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))                             #f(x) = 1 / (1 + e^(-x))

class Neuron:
    def __init__(self, weights, bias):                      # Neuron, [w1, w2], b
        self.weights = weights                              # w1, w2
        self.bias = bias                                    # b
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias    # y = (w . x) + b 
        return sigmoid(total)                               # S(y)

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden later with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])                          # w1 = 0, w2 = 1
        bias = 0                                            # b = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1

network = NeuralNetwork()
x = np.array([2, 3])                                        # x1 = 2, x2 = 3
print(network.feedforward(x))