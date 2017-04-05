import networkx as nx
import numpy as np

s = """
0 1 1 0 0
1 0 1 1 0
1 1 0 1 1
0 1 1 0 1
0 0 1 1 0
""".strip().replace("\n", ";").replace(" ", " , ")

A = np.array(np.matrix(s))
G = nx.from_numpy_matrix(A)

# Part B

s = """
0 1 1 0 0 0 1 1
1 0 1 1 0 0 0 1
1 1 0 1 1 0 0 0
0 1 1 0 1 1 0 0
0 0 1 1 0 1 1 0
0 0 0 1 1 0 1 1
1 0 0 0 1 1 0 1
1 1 0 0 0 1 1 0
""".strip().replace("\n", ";").replace(" ", " , ")

A = np.array(np.matrix(s))
G = nx.from_numpy_matrix(A)

print(nx.diameter(G))