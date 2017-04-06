from math import factorial
import matplotlib.pyplot as plt


def erlang(n, rho):
    def form(i):
        return rho ** i / factorial(i)

    top = form(n)
    bottom = sum(form(i) for i in range(n + 1))

    return top / bottom


if __name__ == '__main__':

    ns = list(range(1, 200))
    prob = [erlang(i, 3 * i) for i in ns]

    plt.figure(figsize=(16, 9))
    plt.plot(ns, prob)
    plt.title("Load and Erlang Formula")
    plt.xlabel('Load and Capacity Multiple')
    plt.ylabel('Probability of Congestion')
    plt.show()

