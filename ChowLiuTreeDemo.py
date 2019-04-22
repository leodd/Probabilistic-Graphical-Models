from ChowLiuTree import build_tree
import numpy as np
import networkx as nx
from nx_tree import hierarchy_pos
import matplotlib.pyplot as plt


data = np.loadtxt('Data/mushroom_train.data', delimiter=',', dtype='<U1')

G = build_tree(data)

pos = hierarchy_pos(G, 0)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
