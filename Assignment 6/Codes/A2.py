from math import factorial as fact


def nCr(n, r):
    return fact(n) // fact(r) // fact(n - r)


def find_NS(ND, gamma=0.01, p=0.1):
    ND += 1

    inc = 1
    while 1:
        prob = 0.
        guess = ND + inc
        for i in range(ND, guess):
            prob += nCr(guess, i) * (p ** i) * ((1 - p) ** (guess - i))

        if prob > gamma:
            return guess - 1
        else:
            inc += 1


if __name__ == '__main__':
    Cs = 10, 20, 30
    for C in Cs:
        print(find_NS(C))
