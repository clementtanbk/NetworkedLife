# Contagion Model
import numpy as np

G = """
0 1 1 0 0 0 1 0
1 0 0 1 0 0 0 1
1 0 0 1 1 1 0 0
0 1 1 0 1 1 0 0
0 0 1 1 0 1 1 0
0 0 1 1 1 0 0 1
1 0 0 0 1 0 0 1
0 1 0 0 0 1 1 0
""".strip().replace('\n', ';').replace(' ', ',')

A = np.array(np.matrix(G))
N = len(A)
p = 0.3

# Part A
x = np.zeros(N)
x[0] = 1

prev = np.zeros(N)
while not all(prev == x):
    prev = x.copy()
    x += (x.dot(A) / A.sum(axis=1)) > p
    x = (x >= 1).astype(int)

# Part B
x = np.zeros(N)
x[2] = 1

prev = np.zeros(N)
while not all(prev == x):
    prev = x.copy()
    x += (x.dot(A) / A.sum(axis=1)) > p
    x = (x >= 1).astype(int)
