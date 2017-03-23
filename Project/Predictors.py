import numpy as np


def softmax(x: np.array):
    values = np.exp(x)
    return values / np.sum(values, axis=1, keepdims=True)


def linear_regression():
    pass


def get_predictors() -> (dict, dict):
    """
    Get available predictors
    :return: predictors, differentiated predictors
    """
    return {
               'softmax': softmax,
               'linear_reg': linear_regression
           }

def predictors_help():
    return """
Available Predictors
1) softmax - softmax
2) linear_reg - linear regression
    """
