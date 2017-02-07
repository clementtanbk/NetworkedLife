import numpy as np

H = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0]
], dtype=np.float)

H[H.sum(axis=1) == 0, :] += 1
H /= H.sum(axis=1)[:, np.newaxis]

N = H.shape[0]

thetas = [0.1, 0.3, 0.5, 0.85]
Gs = []

for t in thetas:
    Gs.append((t, t * H + (1 - t) * np.ones(H.shape) / N))

pis = []
for t, G in Gs:
    pi = np.ones((1, N)) / N
    for _ in range(100):
        pi = pi.dot(G)

    pis.append((t, pi.ravel()))

for t, pi in pis:
    print("theta = %.2f, %s" % (t, np.round(pi, 4)))
