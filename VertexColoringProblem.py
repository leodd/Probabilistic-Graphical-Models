from Graph import *
from BP import BP
from MCMC import MCMC
from math import e
import numpy as np


class EdgePotential(Potential):
    def __init__(self):
        Potential.__init__(self)

    def get(self, parameters):
        return 1 if parameters[0] != parameters[1] else 0


class NodePotential(Potential):
    def __init__(self, weights):
        Potential.__init__(self)
        self.weights = weights

    def get(self, parameters):
        return e ** self.weights[parameters[0]]


def build_graph(A, w):
    n = len(A)

    domain = Domain(tuple(range(len(w))))
    edge_potential = EdgePotential()
    node_potential = NodePotential(w)

    rvs = list()
    factors = list()

    for i in range(n):
        rv = RV(domain, value=None)
        rvs.append(rv)
        factors.append(
            F(node_potential, (rv,))
        )

    for i in range(n):
        for j in range(n):
            if i < j and A[i, j] == 1:
                factors.append(
                    F(edge_potential, (rvs[i], rvs[j]))
                )

    return Graph(rvs, factors)


def prob(A, w, its, max_prod=False):
    n = len(A)

    g = build_graph(A, w)
    rvs = g.rvs

    bp = BP(g, max_prod=max_prod)
    bp.run(iteration=its)

    res = np.zeros((n, len(w)))
    for i in range(n):
        p = bp.prob(rvs[i])
        for k, color in enumerate(w):
            res[i, k] = p[k]
    return res


def sumprod(A, w, its):
    n = len(A)

    g = build_graph(A, w)

    bp = BP(g)
    bp.run(iteration=its)

    return bp.partition()


def maxprod(A, w, its):
    n = len(A)

    g = build_graph(A, w)
    rvs = g.rvs

    bp = BP(g, max_prod=True)
    bp.run(iteration=its)

    x = list()
    for i in range(n):
        x.append(bp.map(rvs[i]))

    return x


def gibbs(A, w, burnin, its):
    n = len(A)

    g = build_graph(A, w)
    rvs = g.rvs

    mcmc = MCMC(g)
    mcmc.run(iteration=its, burnin=burnin)

    res = np.zeros((n, len(w)))
    for i in range(n):
        p = mcmc.prob(rvs[i])
        for k, color in enumerate(w):
            res[i, k] = p[k]
    return res


def generate_samples(A, w, burnin, its):
    n = len(A)

    g = build_graph(A, w)
    rvs = g.rvs

    mcmc = MCMC(g)
    mcmc.run(iteration=its, burnin=burnin)

    res = np.zeros((n, its + 1))
    for i in range(n):
        res[i, :] = mcmc.state[rvs[i]]

    return res


def colormle(A, samples):
    n, m = samples.shape

    # counting the number of distinct color
    color = np.unique(samples)

    d = len(color)
    w = [0] * d

    step_size = 1 / m
    its = 100
    for _ in range(its):
        p = prob(A, w, 10)

        g = [0] * d
        for k in range(m):
            for i in range(n):
                for color_idx in range(d):
                    g[color_idx] += (1 if samples[i, k] == color[color_idx] else 0) - p[i, color_idx]

        # update w[color_idx]
        for color_idx in range(d):
            w[color_idx] += g[color_idx] * step_size
    
    return w


def colorem(A, L, samples):
    n, m = samples.shape

    # counting the number of distinct color
    color = np.unique(samples)

    d = len(color)
    w = [0] * d

    step_size = 1 / m
    its = 20
    for _ in range(its):
        # E step
        p_mis_table = list()
        for k in range(m):
            p_mis_table.append(prob_with_evidence(A, w, L, samples[:, k], 10))

        # M step
        for _ in range(10):
            p = prob(A, w, 10)

            g = [0] * d
            for k in range(m):
                p_mis = p_mis_table[k]

                for i in range(n):
                    if L[i] == 0:
                        for color_idx in range(d):
                            g[color_idx] += (1 if samples[i, k] == color[color_idx] else 0) - p[i, color_idx]
                    else:
                        for i_color_idx in range(d):
                            for color_idx in range(d):
                                g[color_idx] += p_mis[i, i_color_idx] * \
                                                ((1 if color[i_color_idx] == color[color_idx] else 0) - p[i, color_idx])

            # update w[color_idx]
            for color_idx in range(d):
                w[color_idx] += g[color_idx] * step_size

    return w


def prob_with_evidence(A, w, L, sample, its, max_prod=False):
    n = len(A)

    domain = Domain(tuple(range(len(w))))
    edge_potential = EdgePotential()
    node_potential = NodePotential(w)

    rvs = list()
    factors = list()

    for i in range(n):
        rv = RV(domain, value=(int(sample[i]) if L[i] == 0 else None))
        rvs.append(rv)
        factors.append(
            F(node_potential, (rv,))
        )

    for i in range(n):
        for j in range(n):
            if i < j and A[i, j] == 1:
                factors.append(
                    F(edge_potential, (rvs[i], rvs[j]))
                )

    g = Graph(rvs, factors)

    bp = BP(g, max_prod=max_prod)
    bp.run(iteration=its)

    res = np.zeros((n, len(w)))
    for i in range(n):
        p = bp.prob(rvs[i])
        for k in range(len(w)):
            res[i, k] = p[k]
    return res
