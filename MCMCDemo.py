from VertexColoringProblem import gibbs
import numpy as np


A = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
)

w = [1, 2]

# you can even print the distribution of each variables
ps = gibbs(A, w, burnin=10, its=1000)
for p in ps:
    print(p)
