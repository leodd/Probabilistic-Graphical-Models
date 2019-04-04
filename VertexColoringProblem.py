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


def prob(A, w, its, max_prod=True):
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

    g = Graph(rvs, factors)

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

    g = Graph(rvs, factors)

    bp = BP(g)
    bp.run(iteration=its)

    return bp.partition()


def maxprod(A, w, its):
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

    g = Graph(rvs, factors)

    bp = BP(g, max_prod=True)
    bp.run(iteration=its)

    x = list()
    for i in range(n):
        x.append(bp.map(rvs[i]))

    return x


def gibbs(A, w, burnin, its):
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

    g = Graph(rvs, factors)

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

    g = Graph(rvs, factors)

    mcmc = MCMC(g)
    mcmc.run(iteration=its, burnin=burnin)

    for 

    return 


def colormle(A, samples):
    n, m = samples.shape

    # counting the number of distinct color
    color = np.unique(samples)
    print(color)

    d = len(color)
    w = [1] * d

    step_size = 0.4
    its = 200
    for _ in range(its):
        p = prob(A, w, 10)

        for color_idx in range(d):
            # compute the gradient w.r.t. w[color_idx]
            g = 0
            for i in range(n):
                for k in range(m):
                    g += (1 if samples[i, k] == color[color_idx] else 0) - p[i, color_idx]
            
            # update w[color_idx]
            w[color_idx] += g * step_size
    
    return w
