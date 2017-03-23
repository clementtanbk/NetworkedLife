import pandas as pd
import numpy as np


def get_train_test_split(raw=False) -> (np.array, np.array, np.array, np.array):
    """
    :param raw: if True, training dataset will be converted to categorical matrix
    :return: x_train, x_test, y_train, y_test
    """

    train = pd.read_csv('training.csv', names=['Movie', 'User', 'Rating'])
    test = pd.read_csv('validation.csv', names=['Movie', 'User', 'Rating'])

    if raw:
        train, test = train.as_matrix(), test.as_matrix()
        x_train, x_test = train[:, :-1], train[:, -1]
        y_train, y_test = test[:, :-1], test[:, -1]

    else:
        n_train, n_test = len(train), len(test)  # Number of data points
        n_movies = len(train['Movie'].unique())
        n_users = len(train['User'].unique())

        cols = n_movies + n_users

        x_train, x_test = np.zeros((n_train, cols)), np.zeros((n_test, cols))
        for i, (j, k) in enumerate(train.ix[:, ('Movie', 'User')].as_matrix()):
            x_train[i, j] = 1
            x_train[i, n_movies + k] = 1

        for i, (j, k) in enumerate(test.ix[:, ('Movie', 'User')].as_matrix()):
            x_test[i, j] = 1
            x_test[i, n_movies + k] = 1

        y_train = train.Rating.as_matrix()
        y_test = test.Rating.as_matrix()

    return x_train, y_train, x_test, y_test


def RMSE(pred: np.array, true: np.array):
    return np.sqrt(np.mean((pred - true) ** 2))


def accuracy(pred: np.array, true: np.array):
    return np.mean(pred == true)
