from sklearn import datasets
import matplotlib.pyplot as plt


def test_moons(samples=1000, noise=0.2, show_samples=False):
    x, y = datasets.make_moons(samples, noise=noise)

    if show_samples:
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()
