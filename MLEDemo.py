from VertexColoringProblem import colormle
import numpy as np


A = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

samples = np.array([
    [1, 3, 3, 1, 2],
    [2, 2, 1, 3, 1],
    [1, 3, 2, 2, 3]
])



print(colormle(A, samples))
