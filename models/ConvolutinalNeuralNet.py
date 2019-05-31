import numpy as np


def tensorize(x):
    return np.squeeze(np.asfarray(x))


# Define a Layer Parent class. Will be used by Relu Layers, SoftMax Layers, and FC Layers
class Layer:

    def __init__(self, parms, f, dfdx):
        self.parms = [tensorize(p) for p in parms]
        self.f = f
        self.dfdx = dfdx
        self.x = None

    def reset(self, r=None):
        self.x = None

    def getWeights(self):
        if len(self.parms) == 0:
            return []
        else:
            return np.concatenate([p.flatten() for p in self.parms])

    def setWeights(self, w):
        if len(w) > 0:
            w = tensorize(w)
            for k in range(len(self.parms)):
                s = self.parms[k].shape
                n = 1 if len(s) == 0 else np.prod(s)
                self.parms[k] = np.reshape(w[:n], s)
                w = w[n:]

    def dfdw(self):
        assert self.x is not None, 'dfdw called before f'
        return np.empty((len(self.x), 0))


# Implement the FC Layer. FC Layers looks at what high level features
# most strongly correlate to a particular class and has particular weights
class FCLayer(Layer):

    def __init__(self, V, b):
        V, b = tensorize(V), tensorize(b)

        def f(x):
            self.x = tensorize(x)
            return np.dot(self.parms[0], self.x) + self.parms[1]

        def dfdx():
            assert self.x is not None, 'dfdx called before f'
            return self.parms[0]

        Layer.__init__(self, [V, b], f, dfdx)

    def dfdw(self):
        assert self.x is not None, 'dfdw called before f'
        m, n = self.parms[0].shape
        D = np.zeros((m, m * (n + 1)))
        js, je = 0, n
        for i in range(m):
            D[i][js:je] = self.x
            js, je = js + n, je + n
        D[:, (m * n):] = np.diag(np.ones(m))
        return D

    def __initialWeights(m, n, r=None):
        if r is None:
            r = np.sqrt(2 / m)  # Formula by He et al.
        V = np.random.randn(n, m) * r
        b = np.zeros(n)
        return V, b

    @classmethod
    def ofShape(cls, m, n, r=None):
        V, b = FCLayer.__initialWeights(m, n, r)
        return cls(V, b)

    def reset(self, r=None):
        self.x = None
        n, m = self.parms[0].shape
        V, b = FCLayer.__initialWeights(m, n, r)
        self.parms = [V, b]


# Implement the Relu Layer to be used in the hidden layers
class ReLULayer(Layer):

    def __init__(self):
        def f(x):
            self.x = tensorize(x)
            return (np.maximum(0, self.x))

        def dfdx():
            assert self.x is not None, 'dfdx called before f'
            x_arr = np.atleast_1d(self.x)  # Turn our x into an array if not already
            e = x_arr.size
            J = np.zeros([e, e])
            diagonal = np.array([0.5 * (1 + np.sign(x_arr))])
            if diagonal.shape == (1, 1):
                return (diagonal[0])
            return (J + np.diag(diagonal[0]))

        Layer.__init__(self, [], f, dfdx)


# Define SoftMax Layer, which is used at the end of the network

class SoftmaxLayer(Layer):

    def __softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def __init__(self, n):
        def f(x):
            self.x = tensorize(x)
            return SoftmaxLayer.__softmax(self.x)

        def dfdx():
            assert self.x is not None, 'dfdx called before f'
            s = SoftmaxLayer.__softmax(self.x)
            diagonal = np.diag(np.array([s])[0])

            return (np.subtract(diagonal, np.outer(s, s)))

        Layer.__init__(self, [], f, dfdx)


# Define the Loss Function. This class defines the *Cross Entropy* Loss function as well as the Jacobian of
# The cross entropy loss
class Loss:
    def __init__(self):
        self.small = 1e-8

    def f(self, y, p):
        self.p = tensorize(p)
        py = self.p[int(y)]
        if py < self.small: py = self.small
        return (-np.log(py))

    def dfdx(self, y):
        assert self.p is not None, 'dfdx called before f'
        y = int(y)
        d = np.zeros(len(self.p))
        py = self.p[y]
        if py < self.small: py = self.small
        for i in range(len(self.p)):
            if (i == y):
                d[i] = -1 / (py)
        return (d)


# Lastly, define the NEtwork Class! This class takes parameters that tells it how many FC and Relu layers to
# add, as well as defining the input and output layers

# GetWeights() -> Returns the weights of each layer
# SetWeights() -> Sets weights for each layer
# f() -> Returns the activation function of each layer
# backprop() -> Performs backpropoagation to train the network

class Network:

    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(FCLayer.ofShape(sizes[i], sizes[i + 1]))
            self.layers.append(ReLULayer())
        self.layers.append(SoftmaxLayer(sizes[-1]))
        self.p = None

    def reset(self, r=None):
        for layer in self.layers: layer.reset(r)
        self.p = None

    def getWeights(self):
        return np.concatenate([layer.getWeights() for layer in self.layers])

    def setWeights(self, w):
        for layer in self.layers:
            n = len(layer.getWeights())
            layer.setWeights(w[:n])
            w = w[n:]

    def f(self, x):
        x = tensorize(x)
        for layer in self.layers: x = layer.f(x)
        self.p = x
        return self.p

    def backprop(self, x, y, loss):
        L = loss.f(y, self.f(x))
        g = loss.dfdx(y)
        g = g.reshape((1, len(g)))
        deltan_stor = np.empty((0,))  # Storage for the nth sample to the loss gradient
        for k in range(len(self.layers) - 1, -1, -1):
            delta_n = np.dot(g, self.layers[k].dfdw())
            if delta_n.size > 0:
                deltan_stor = np.concatenate((deltan_stor, delta_n[0]))
            g = np.dot(g, self.layers[k].dfdx())  # delta of loss with respect to x(k-1)
        return ((L, deltan_stor))



# ReadArray Method to read our data in

def readArray(filename):
    with open(filename, 'r') as file:
        X = np.array([[float(a) for a in line.strip().split()] for line in file])
    return X
