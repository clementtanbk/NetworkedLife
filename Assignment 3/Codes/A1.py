import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import os

B = nx.Graph()
B.add_nodes_from([
    (1, {'revenue': 6}),
    (2, {'revenue': 4}),
    (3, {'revenue': 3})
], bipartite=0)

B.add_nodes_from([
    ('A', {'click-rate': 500}),
    ('B', {'click-rate': 300})
], bipartite=1)

set0 = [i[0] for i in B.nodes(True) if i[1]['bipartite'] == 0]
set1 = [i[0] for i in B.nodes(True) if i[1]['bipartite'] == 1]
nodes = {i: j for i, j in B.nodes(True)}

edges = []

for i, j in product(set0, set1):
    revenue = nodes[i]['revenue'] * nodes[j]['click-rate']
    edges.append((i, j, {
        'weight': revenue,
        'bold': False
    }))

B.add_edges_from(edges)
for i, j in nx.max_weight_matching(B).items():
    if i in set1:
        B.get_edge_data(i, j)['bold'] = True

pos = {j: (0, i) for i, j in enumerate(set0)}
pos.update({j: (1, i + 0.5) for i, j in enumerate(set1)})

# Drawing the graph
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
nx.draw_networkx_edges(B, pos,
                       edgelist=[i for i, j in nx.get_edge_attributes(B, 'bold').items() if j],
                       width=bold_width)
nx.draw_networkx_edges(B, pos,
                       edgelist=[i for i, j in nx.get_edge_attributes(B, 'bold').items() if not j])
nx.draw_networkx_edge_labels(B, pos,
                             edge_labels=nx.get_edge_attributes(B, 'weight'),
                             label_pos=0.25)

plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Images', 'Image-1.png'))
