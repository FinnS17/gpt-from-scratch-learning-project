import random
from .engine import Value
import math

class Neuron:
    def __init__(self, nin):
        # create one weight for each input going into this neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # bias term for the neuron
        self.b = Value(0)

    def __call__(self, x):
        # compute weighted sum: w1*x1 + w2*x2 + ... + bias
        # sum(...) starts at self.b so bias is included
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # pass activation through tanh so the neuron becomes nonlinear
        out = act.tanh()
        return out

    def parameters(self):
        # return all learnable parameters of this neuron
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # create 'nout' neurons, each receiving 'nin' inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # feed input x into every neuron in the layer
        # output is a list of neuron outputs
        return [n(x) for n in self.neurons]

    def parameters(self):
        # collect parameters from all neurons inside this layer
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        # nouts = list like [4,4,1] for a 3-layer network
        # build layer sizes: e.g. [3,4,4,1] if nin=3
        sz = [nin] + nouts
        # create all layers: Layer(3,4), Layer(4,4), Layer(4,1)
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # forward pass: push x through each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x  # final output of the network

    def parameters(self):
        # gather all parameters from every layer
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params