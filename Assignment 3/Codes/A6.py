import numpy as np

H = np.eye(4)
pi = np.array([0.5, 0.5, 0, 0])

for i in range(10):
    pi = pi.reshape((4, 1))
    pi = pi.T.dot(H)
