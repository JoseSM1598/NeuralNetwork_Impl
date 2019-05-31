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

class SigmoidLayer(Layer):

    def __sigmoid(self, x):
        #z = np.sum(x*self.getWeights())
        return 1 / (1 + np.exp(-1*x))

    def __init__(self, n):
        def f(x):
            self.x = tensorize(x)
            return SigmoidLayer.__sigmoid(self,self.x)

        def dfdx():
            assert self.x is not None, 'dfdx called before f'
            s = SigmoidLayer.__sigmoid(self,self.x)
            diagonal = np.diag(np.array([s])[0])

            return (np.subtract(diagonal, np.outer(s, s)))

        Layer.__init__(self, [], f, dfdx)

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

    def __init__(self, relu_layers, output_size):
        self.layers = []
        for i in range(relu_layers):
            self.layers.append(ReLULayer())
        self.layers.append(SigmoidLayer(output_size))
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
