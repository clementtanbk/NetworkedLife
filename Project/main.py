import matplotlib.pyplot as plt

from Project.LinearRegression import LinearRegression
from Project import utils


def run_linear_regression():
    x, y, x_test, y_test = utils.get_train_test_split()
    train_error = []
    test_error = []

    alphas = sorted([0] + [10 ** i for i in range(-2, 2)])

    for i in alphas:
        lm = LinearRegression(alpha=i).fit(x, y)
        train_error.append(utils.RMSE(lm.predict(x), y))
        test_error.append(utils.RMSE(lm.predict(x_test), y_test))

    plt.plot(alphas, train_error, label='Train Error')
    plt.plot(alphas, test_error, label='Test Error')

    plt.title("Error Plot")
    plt.xlabel("l2 Penalty")
    plt.ylabel("Error")
    plt.ylim(min(*test_error, *train_error) - 0.25, max(*test_error, *train_error) + 0.25)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    run_linear_regression()
