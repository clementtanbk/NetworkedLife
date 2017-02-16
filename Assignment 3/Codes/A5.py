import networkx as nx
import matplotlib.pyplot as plt
import os

B = nx.Graph()
B.add_nodes_from([1, 2, 3, 4], bipartite=0)

B.add_nodes_from(['A', 'B'], bipartite=1)

set0 = [i[0] for i in B.nodes(True) if i[1]['bipartite'] == 0]
set1 = [i[0] for i in B.nodes(True) if i[1]['bipartite'] == 1]
nodes = {i: j for i, j in B.nodes(True)}

B.add_edges_from([
    (1, 'A', {'weight': 10}),
    (2, 'A', {'weight': 3}),
    (3, 'A', {'weight': 5}),
    (4, 'A', {'weight': 7}),
    (1, 'B', {'weight': 2}),
    (2, 'B', {'weight': 8}),
    (3, 'B', {'weight': 6}),
    (4, 'B', {'weight': 9})
])

pos = {j: (0, i) for i, j in enumerate(reversed(set0))}
pos.update({j: (1, i + 1) for i, j in enumerate(set1)})

node_size = 500
bold_width = 5

nx.draw(B, pos)
nx.draw_networkx_nodes(B, pos,
                       nodelist=set0,
                       node_color='b',
                       node_size=node_size)
nx.draw_networkx_nodes(B, pos,
                       nodelist=set1,
                       node_size=node_size)
nx.draw_networkx_labels(B, pos,
                        labels={i: i for i in B.nodes()},
                        font_color='w')
nx.draw_networkx_edge_labels(B, pos,
                             edge_labels=nx.get_edge_attributes(B, 'weight'),
                             label_pos=0.75)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Images', 'Image-2.png'))

print(nx.max_weight_matching(B))
