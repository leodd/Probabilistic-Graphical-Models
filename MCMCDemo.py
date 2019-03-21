from VertexColoringProblem import gibbs
import numpy as np


# A = np.array(
#     [[0, 1, 0],
#      [1, 0, 1],
#      [0, 1, 0]]
# )

A = np.array(
    [[0, 1, 1, 1],
     [1, 0, 0, 1],
     [1, 0, 0, 1],
     [1, 1, 1, 0]]
)

w = [1, 2, 3, 4]

v = [2 ** 6, 2 ** 10, 2 ** 14, 2 ** 18]

for burnin in v:
    for its in v:
        res = gibbs(A, w, burnin=burnin, its=its)
        print(f'burnin: {burnin}, its: {its}')
        print(res)
        print('---------------------')

