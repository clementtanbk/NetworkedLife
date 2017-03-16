import numpy as np


#TODO add regularizer

def categorical_loss(pred: np.array, true: np.array):
    return - np.sum(true * np.log(pred))


def d_categorical_loss(pred: np.array, true: np.array):
    return pred - true


def get_loss_functions() -> (dict, dict):
    """
    Returns available loss functions and its derivative
    :return:
    """
    return {
               'categorical_loss': categorical_loss
           }, {
               'categorical_loss': d_categorical_loss
           }


def loss_function_help():
    return """
Available Loss Functions:
1) categorical_loss - Categorical Loss Entropy
    """
