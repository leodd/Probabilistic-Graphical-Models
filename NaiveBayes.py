import numpy as np
from collections import Counter


class NaiveBayes:
    def __init__(self):
        self.prior = dict()
        self.theta = list()

    def train(self, Y, X):
        m, d = X.shape

        self.prior = dict()
        self.theta = list()
        for _ in range(d):
            self.theta.append(dict())

        y_value_count = Counter(Y)

        for y, y_num in y_value_count.items():
            self.prior[y] = y_num / m
            y_idx = Y == y

            for i in range(d):
                p = self.theta[i]
                x_value_count = Counter(X[y_idx, i])

                for x, x_num in x_value_count.items():
                    p[(y, x)] = x_num / y_num

    def train_with_uniform_dirichlet_prior(self, Y, X, domain, scale):
        m, d = X.shape

        self.prior = dict()
        self.theta = list()
        for _ in range(d):
            self.theta.append(dict())

        y_value_count = Counter(Y)

        for y, y_num in y_value_count.items():
            self.prior[y] = (y_num + scale / len(y_value_count)) / (m + scale)
            y_idx = Y == y

            for i in range(d):
                p = self.theta[i]
                x_value_count = Counter(X[y_idx, i])

                a_xy = scale / (len(domain[i]) + len(y_value_count))
                a_y = scale

                for x in domain[i]:
                    p[(y, x)] = (x_value_count.get(x, 0) + a_xy) / (y_num + a_y)

    def predict(self, X):
        m, d = X.shape

        res = np.empty(m, dtype='<U1')

        for k in range(m):
            max_p = -1
            for y in self.prior.keys():
                p = self.prior[y]
                for i in range(d):
                    p *= self.theta[i].get((y, X[k, i]), 0)
                if p > max_p:
                    res[k] = y
                    max_p = p

        return res
