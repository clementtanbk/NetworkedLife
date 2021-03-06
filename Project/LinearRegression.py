import numpy as np
from numpy import array
import numpy.linalg as la

from Project import utils


class LinearRegression:
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.r_bar = None
        self.b = None

    def fit(self, x: array, y: array):
        self.r_bar = r_bar = y.mean()

        y = (y - r_bar).reshape((len(y), 1))
        n = len(x.T.dot(x))  # number of factors
        self.b = la.inv(x.T.dot(x) + self.alpha * np.identity(n)).dot(x.T).dot(y)

        return self

    def predict(self, x: array):
        """
        Predicts results given input array
        :param x: input array
        :return: y, predictions
        """
        if self.b is None:
            raise RuntimeError("Model is not yet trained. Fit some data first")

        return np.dot(x, self.b) + self.r_bar

    def get_params(self):
        """
        Returns the trained model parameters. We can take r_bar to be the bias term.
        :return: model coefficients, bias, reg
        """

        if self.b is None:
            raise RuntimeError("Model is not yet trained, Fit some data first")

        return self.b, self.r_bar, self.alpha
