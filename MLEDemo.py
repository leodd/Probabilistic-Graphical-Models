from VertexColoringProblem import colormle, colorem, generate_samples
import numpy as np


A = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

# samples = np.array([
#     [1, 3, 3, 3],
#     [2, 2, 1, 2],
#     [1, 3, 2, 3]
# ])

samples = generate_samples(A, [1, 1, 1], 20, 1000)

L = [0, 1, 0]

print(colormle(A, samples))
# print(colorem(A, L, samples))
