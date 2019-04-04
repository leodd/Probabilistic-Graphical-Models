from VertexColoringProblem import prob, sumprod, maxprod
import numpy as np

A = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
)

w = [1, 2]

print(sumprod(A, w, 10))
print(maxprod(A, w, 10))

# you can even print the distribution of each variables
print(prob(A, w, 10, max_prod=False))
