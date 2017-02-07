import numpy as np

iterations = 200


H = np.array([
    [1, 0],
    [1 / 3, 2 / 3]
], dtype=np.float)

N = H.shape[0]
t = 0.85

pi = np.ones((1, N)) / N

G = t * H + (1 - t) * np.ones(H.shape) / N

for _ in range(iterations):
    pi = pi.dot(G)

pi = pi.ravel()
print(np.round(pi, 4))

# Part 2
N1 = 2
H1 = np.eye(N1, dtype=np.float)
G1 = t * H1 + (1 - t) * np.ones(H1.shape) / N1

N2 = 3
H2 = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
], dtype=np.float)

H2 += (np.ones(N2, dtype=np.float) * (H2.sum(axis=1) == 0))[:, np.newaxis]
H2 /= H2.sum(axis=1)[:, np.newaxis]
G2 = t * H2 + (1 - t) * np.ones(H2.shape) / N2

pi1 = np.ones(N1) / N1
pi2 = np.ones(N2) / N2

for _ in range(iterations):
    pi1 = pi1.dot(G1)
    pi2 = pi2.dot(G2)

# Part 3
left = pi[0] * pi1
right = pi[1] * pi2

pi_ = np.concatenate((left, right))

print(np.round(pi_, 4))
