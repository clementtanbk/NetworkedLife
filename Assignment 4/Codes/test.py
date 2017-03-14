import numpy as np
import networkx as nx
from itertools import product

G = r"""
0 1 1 1 0 0 0 0
1 0 0 1 1 0 0 0
1 0 0 0 0 1 1 1
1 1 0 0 1 0 0 0
0 1 0 1 0 0 0 0
0 0 1 0 0 0 1 0
0 0 1 0 0 1 0 1
0 0 1 0 0 0 1 0
""".strip().replace('\n', ';').replace(' ', ',')

G = np.array(np.matrix(G))
N = len(G)
G = nx.from_numpy_matrix(G)
G = nx.relabel_nodes(G, {i: i + 1 for i in range(N)})

link = (1, 2)
M = np.zeros((N, N))
D = np.zeros((N, N))

for s, t in product(range(1, N + 1), range(1, N + 1)):
    if s == t:
        continue
    count = 0

    for path in nx.all_shortest_paths(G, s, t):
        links = set(zip(path, path[1:]))
        if link in links:
            count += 1
    M[s - 1, t - 1] = count
    D[s - 1, t - 1] = len(list(nx.all_shortest_paths(G, s, t)))

D += np.eye(N)
M = M + M.T
T = M / D
