import numpy as np
import networkx as nx
from collections import defaultdict


def marginal_distribution(data, i):
    m, _ = data.shape

    distribution = defaultdict(float)
    s = 1 / m
    for x in data:
        distribution[x[i]] += 1
    return distribution


def join_distribution(data, i, j):
    distribution = defaultdict(float)
    s = 1 / len(data)
    for x in data:
        distribution[(x[i], x[j])] += s
    return distribution


def mutual_information(data, i, j):
    pi = marginal_distribution(data, i)
    pj = marginal_distribution(data, j)
    pij = join_distribution(data, i, j)
    information = 0
    for a, b in pij:
        information += pij[(a, b)] * np.log(pij[(a, b)] / (pi[a] * pj[b]))
    return information


def build_tree(data):
    _, d = data.shape

    G = nx.Graph()

    for i in range(d):
        G.add_node(i)
        for j in range(i):
            G.add_edge(i, j, weight=-mutual_information(data, i, j))

    return nx.minimum_spanning_tree(G)
