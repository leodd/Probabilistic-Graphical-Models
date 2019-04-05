from VertexColoringProblem import colormle, generate_samples
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

samples = generate_samples(A, [1, 2, 3], 20, 1000)

print(colormle(A, samples))
