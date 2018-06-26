import numpy as np

# Eigenvalues
M = np.array([[0.1, 0.1, 0.1, 0.7],
              [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.7, 0.1, 0.1],
              [0.1, 0.1, 0.7, 0.1]])
vals, vecs = np.linalg.eig(M)

M = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]])


M = np.array([[0.1, 0.7, 0.1, 0.1],
              [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.7],
              [0.1, 0.1, 0.7, 0.1]])
det = np.linalg.det(M)
print("Det", det)

vals, vecs = np.linalg.eig(M)

print(vals)
print(vecs)
