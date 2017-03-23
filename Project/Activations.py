import numpy as np


def relu(x: np.array):
    return (x > 0) * x


def d_relu(x: np.array):
    return (x > 0).astype(np.int)


def sigmoid():
    return


def d_sigmoid():
    return


def get_activations() -> (dict, dict):
    """
    Returns the available activation functions and their derivative functions
    :return: (dict, dict)
    """
    return {
               'relu': relu,
               'sigmoid': sigmoid
           }, {
               'relu': d_relu,
               'sigmoid': d_sigmoid
           }


def activation_help():
    return """
Available action functions:
1) relu - rectificed linear unit
2) sigmoid - sigmoid
    """
