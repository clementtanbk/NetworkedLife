from inspect import isfunction

import numpy as np
import numpy.random as rng
import Project.utils as utils

from Project.Activations import get_activations, activation_help
from Project.LossFunctions import get_loss_functions, loss_function_help
from Project.Predictors import get_predictors, predictors_help


class RBM:
    def __init__(self, learning_rate=0.1, init="random", random_state=None, loss_function="categorical_loss",
                 predictor="softmax"):

        # initialization variables
        self.init = init
        self._seed = random_state

        # neural network hidden layer variables
        self._acts, self._dacts = get_activations()  # activation functions and its derivative that are available
        self._hidden_layer_shape = []  # stores the number of nodes
        self._bias = []
        self._weights = []
        self._activation = []
        self._dropout = []

        # training variables
        self._lr = learning_rate

        self._loss_function = loss_function

        self._predictor = predictor.lower()

        self._errors = []

        # backprop
        self._zs = []
        self._as = []

    def _error_rate(self):
        pass

    def _make_layers(self, x: np.array, y: np.array):
        """
        Forms the hidden layer's weights matrix automatically
        :return: self
        """

        if len(self._activation) == 0:
            raise RuntimeError("RBM layers not specified. Use rbm.add_layer(...) to add some layers first")

        rng.seed(self._seed)

        # Make the hidden layers
        _, cols = x.shape
        for num_nodes in self._hidden_layer_shape:
            size = (cols, num_nodes)
            i = num_nodes ** -0.5  # random initializer range
            self._weights.append(rng.uniform(-i, i, size))  # init random weights
            self._bias.append(rng.uniform(-i, i, (1, num_nodes)))
            cols = num_nodes

        # Makes the final layer
        cats = y.shape[1]  # number  of unique categories
        i = cols ** -0.5
        self._weights.append(rng.uniform(-i, i, (cols, cats)))
        self._bias.append(rng.uniform(-i, i, (1, cats)))

        return self

    def _forward_pass(self, x):

        acts = self._acts  # dict: available activation functions. e.g. {'relu': relu_func(x) ... }
        self._zs = []  # matrix: store non-activated value of each node
        self._as = []  # matrix: store activated value of each ndoe

        a = x
        for i, name in enumerate(self._activation):
            func = acts[name]  # activation for the layer
            z = np.dot(a, self._weights[i]) + self._bias[i]
            a = func(z)
            self._zs.append(z)
            self._as.append(a)

        predictor = get_predictors()[self._predictor]
        pred = predictor(np.dot(self._as[-1], self._weights[-1]) + self._bias[-1])
        return pred

    def _backprop(self, pred, true):
        _, d = get_loss_functions()
        d_loss_function = d[self._loss_function]
        deltas = [d_loss_function(pred, true)]
        for i, name in enumerate(self._activation[::-1]):
            index = -i - 1
            func = self._dacts[name]
            weights = self._weights[index]
            d = weights.dot(deltas[-1].T).T * func(self._zs[index])
            deltas.append(d)

        n = len(pred)  # number of samples
        for i, d in enumerate(deltas[:-1]):
            index = -i - 1
            self._weights[index] -= self._lr * np.dot(self._as[index].T, d) / n
            self._bias[index] -= self._lr * d.sum(axis=0, keepdims=True) / n

        return self

    def add_layer(self, nodes, activation='relu', dropout=None):
        """
        Sets the layer information
        :param nodes: number of nodes for layer
        :param activation: activation function for layer
        :param dropout: dropout fraction, must be between 0 and 1
        :return: self
        """
        self._hidden_layer_shape.append(nodes)

        # sets activation function for layer

        act_name = activation.lower()
        if act_name not in self._acts.keys():
            raise ValueError('Activation function not recognized')

        self._activation.append(act_name)

        # TODO add dropout
        self._dropout.append(dropout)

        return self

    def set_loss_function(self, loss_function: str):
        """
        Sets the loss function.
        :param loss_function: name of loss function
        :return: self
        """

        f, d = get_loss_functions()
        n = loss_function.lower()

        if n not in f.keys():
            raise ValueError("Unrecognized loss function. Use rbm.help() to find valid functions")

        self._loss_function = n

        return self

    def set_predictor(self, predictor: str):
        """
        Sets the predictor function
        :param predictor: name of predictor function
        :return: self
        """

        f = get_predictors()
        n = predictor.lower()

        if n not in f.keys():
            raise ValueError("Unrecognized predictor function. Use rbm.help() to find valid functions")

        self._predictor = n

        return self

    def fit(self, x: np.array, y: np.array, epochs=500, categorize_y=True, verbose=True):
        """
        Trains the neural network
        :param x: input matrix
        :param y: ideal test vector
        :param epochs: number of times to run training
        :param categorize_y: converts categorical *y* vector into 1-hot matrix, use if *y* is not 1-hot encoded and
                             *y* is a category. If regression, set this to False.
        :return: self
        """

        if categorize_y:
            ny = np.zeros((len(y), len(np.unique(y))))
            for i, j in enumerate(y):
                ny[i, j - 1] = 1
            y = np.copy(ny)
            del ny

        self._make_layers(x, y)

        f, _ = get_loss_functions()
        loss_function = f[self._loss_function]
        for e in range(epochs):
            pred = self._forward_pass(x)
            self._backprop(pred, y)

            error = loss_function(pred, y)
            self._errors.append(error)
            if verbose and (e + 1) % 25 == 0:
                print("Epoch {0}: {1:.3f}".format(e + 1, error))

        return self

    def predict(self, x):
        pred = self._forward_pass(x)
        return np.argmax(pred, axis=1)

    def describe(self):
        message = """
Neural Network Stats

Number of hidden layers: {0}
""".format(len(self._activation) + 1)
        count = 1
        for nodes, act in zip(self._hidden_layer_shape, self._activation):
            message += """
Layer {0}:
Num Nodes:  {1}
Activation: {2}
""".format(count, nodes, act)
            count += 1

        message += """
Output Layer:
Activation:     {0}
Loss Function:  {1}
""".format(self._predictor, self._loss_function)

        print(message)
        return self

    def help(self):
        print("""
Help Menu

Methods:
add_layer           - Adds a hidden layer. Read docs
describe            - Describes the neural network
fit                 - Trains the neural network. Read docs
predict             - Predicts the labels
set_loss_function   - Sets loss function from available loss functions
set_predictor       - Sets the predictor function from available predictors

{activation}
{predictor}
{loss}
        """.format(activation=activation_help(),
                   predictor=predictors_help(),
                   loss=loss_function_help()))
        return self


if __name__ == '__main__':
    x, y, x_test, y_test = utils.get_train_test_split()

    rbm = RBM(0.5, random_state=888)
    rbm.add_layer(64)
    rbm.add_layer(48)
    rbm.add_layer(48)
    rbm.add_layer(32)
    rbm.add_layer(32)
    # rbm.describe()
    rbm.fit(x, y, epochs=10000, verbose=False)
    preds = rbm.predict(x_test)
    print(utils.accuracy(preds, y_test))
    # rbm.help()
