import numpy as np

A = np.array([[0, 1], [1, 0]])
print(A)
vals, vecs = np.linalg.eig(A)

# print(-1 - np.sqrt(3))

print(vals)
print(vecs.transpose())
# C = vecs

# print(C)
# Cinv = np.linalg.inv(C)
#
# D = Cinv @ A @ C
#
# print(D)
