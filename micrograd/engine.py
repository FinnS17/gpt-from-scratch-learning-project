import math
import random

class Value:
    def __init__(self, data, _children=(), _op=""):
        # 'data' is the actual number we store
        self.data = data
        # gradient of this value (filled in during backprop)
        self.grad = 0.0

        # function to compute gradients of parents (set later)
        self._backward = lambda: None

        # references to previous Values that created this one
        self._prev = set(_children)

        # the operation that produced this Value 
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # ensure 'other' is a Value object
        other = other if isinstance(other, Value) else Value(other)

        # forward: create new Value holding the sum
        out = Value(self.data + other.data, (self, other), '+')

        # backward: derivative of (a + b) wrt both inputs is 1
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # ensure 'other' is a Value
        other = other if isinstance(other, Value) else Value(other)

        # forward: multiply numbers
        out = Value(self.data * other.data, (self, other), '*')

        # backward: derivative of a*b is:
        # da = b, db = a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # forward: apply tanh
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        # backward: derivative of tanh is (1 - tanh^2)
        def _backward():
            self.grad += (1 - t*t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        # forward: exp
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        # backward: derivative of exp(x) is exp(x)
        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # backprop using topological sort
        topo = []
        visited = set()

        # recursively build graph ordering
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # start gradient at output = 1
        self.grad = 1.0

        # go backwards and apply each node's _backward()
        for node in reversed(topo):
            node._backward()