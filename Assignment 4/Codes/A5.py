# This script is to test the information centrality

import numpy as np
import numpy.linalg as la
import networkx as nx

# G = r"""
# 0 1 1 1 0
# 1 0 0 0 1
# 1 0 0 1 1
# 1 0 1 0 1
# 0 1 1 1 0
# """.strip().replace("\n", ";").replace(" ", " , ")

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

G = np.array(np.matrix(G))
N = len(G)

A = np.ones((N, N)) - G + np.eye(N) * G.sum(axis=1) # - np.eye(N)
C = la.inv(A)

I = 1 / (C.diagonal() + C.diagonal().sum() - 2 * C.sum(axis=1))
print(I)

v = np.array(
    list(nx.information_centrality(nx.from_numpy_matrix(G)).values())
)
