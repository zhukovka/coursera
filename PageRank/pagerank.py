import numpy as np
import numpy.linalg as la

L = np.array([[0, 1 / 2, 1 / 3, 0, 0, 0],
              [1 / 3, 0, 0, 0, 1 / 2, 0],
              [1 / 3, 1 / 2, 0, 1, 0, 1 / 2],
              [1 / 3, 0, 1 / 3, 0, 1 / 2, 1 / 2],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 1 / 3, 0, 0, 0]])


# First let's set up our initial vector,  r
# r = 100 * np.ones(6) / 6  # Sets up this vector (6 entries of 1/6 × 100 each)
# lastR = r
# r = L @ r
# i = 0
# while la.norm(lastR - r) > 0.01:
#     lastR = r
#     r = L @ r
#     i += 1
# print(str(i) + " iterations to convergence.")
#
# print(r)


# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
def pageRank(linkMatrix, d):
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1 - d) / n * np.ones([n, n])  # np.ones() is the J matrix, with ones for each entry.
    r = 100 * np.ones(n) / n  # Sets up this vector (6 entries of 1/6 × 100 each)
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")
    print(r)

    return r


pageRank(L, 1)
