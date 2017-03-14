import networkx as nx
import numpy as np
import numpy.linalg as la
from itertools import product
import matplotlib.pyplot as plt

矩阵字符串 = r"""
0 1 1 1 0
1 0 0 0 1
1 0 0 1 1
1 0 1 0 1
0 1 1 1 0
""".strip().replace("\n", ";").replace(" ", " , ")

矩阵 = np.array(np.matrix(矩阵字符串))
N = len(矩阵)

网络 = nx.from_numpy_matrix(矩阵)
网络 = nx.relabel_nodes(网络, {i: i + 1 for i in range(len(矩阵))})

# 程度中心性 is degree centrality. By default it's fractional
程度中心性 = nx.degree_centrality(网络)

# Multiply by (total nodes - 1) to get number of connections
for i, j in 程度中心性.items():
    程度中心性[i] = j * (N - 1)

# 接近中心性 is closeness centrality
接近中心性 = {i + 1: round((N - 1) / j.sum(), 3) for i, j in
         enumerate(np.array(nx.algorithms.floyd_warshall_numpy(网络)))}

# 接近中心性 will be the same as nx.closeness_centrality(网络)

# nx.eigenvector_centrality(网络)
特征向量中心性 = np.random.random(len(矩阵))
l, _ = la.eig(矩阵)
l_largest = np.max(l)
for _ in range(400):
    特征向量中心性 = 特征向量中心性.dot(矩阵) / l_largest

# Normalized 特征向量中心性
特征向量中心性 = {i + 1: j / la.norm(特征向量中心性) for i, j in enumerate(特征向量中心性)}

# Betweenness
中介中心性 = {}
for n in range(2, 4):
    count = 0
    total = 0
    for i, j in product(range(1, 6), range(1, 6)):
        if i == n or j == n or i == j:
            continue
        for path in nx.all_shortest_paths(网络, i, j):
            if n in path:
                count += 1
            total += 1

    中介中心性[n] = count / total / 2

# nx.betweenness_centrality(网络) different from 中介中心性

# Link betweenness
链接中心性 = {}
for link in [(3, 4), (2, 5)]:
    M = np.zeros((N, N))
    D = np.zeros((N, N))

    for s, t in product(range(1, N + 1), range(1, N + 1)):
        if s == t:
            continue
        count = 0

        for path in nx.all_shortest_paths(网络, s, t):
            links = set(zip(path, path[1:]))
            if link in links:
                count += 1
        M[s - 1, t - 1] = count
        D[s - 1, t - 1] = len(list(nx.all_shortest_paths(网络, s, t)))

    D += np.eye(N)
    M = M + M.T
    链接中心性[link] = (M / D).sum() / 2

# Getting all the data
message = """
PART A
程度中心性:
{程度中心性}

接近中心性:
{接近中心性}

特征向量中心性:
{特征向量中心性}


PART B
中介中心性
{中介中心性}

PART C:
链接中心性
{链接中心性}
""".format(
    程度中心性=程度中心性,
    接近中心性=接近中心性,
    特征向量中心性=特征向量中心性,
    中介中心性=中介中心性,
    链接中心性=链接中心性)

print(message)
# Draw out graph
pos = {
    1: (0, 1),
    2: (-1, 0),
    3: (0, 0),
    4: (1, 0),
    5: (0, -1)
}

labels = {i: i for i in range(1, N + 1)}
nx.draw(网络, pos)
nx.draw_networkx_labels(网络, pos, labels)
plt.show()