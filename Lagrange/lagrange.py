from scipy import optimize
import numpy as np


# First we define the functions,
def f(x, y):
    return np.exp(-(2 * x * x + y * y - x * y) / 2)


def g(x, y):
    return x * x + 3 * (y + 1) ** 2 - 1


# Next their derivatives,
def dfdx(x, y):
    return 1 / 2 * (-4 * x + y) * f(x, y)


def dfdy(x, y):
    return 1 / 2 * (x - 2 * y) * f(x, y)


def dgdx(x, y):
    return 2 * x


def dgdy(x, y):
    return 6 * (y + 1)


# Next let's define the vector, \nabla\ ∇L,
# that we are to find the zeros of; we'll call this “DL” in the code.
# Then we can use a pre-written root finding methodin scipy to solve.

def DL(xyλ):
    [x, y, λ] = xyλ
    return np.array([
        dfdx(x, y) - λ * dgdx(x, y),
        dfdy(x, y) - λ * dgdy(x, y),
        - g(x, y)
    ])


# Here, the first two elements of the array are the x and y coordinates that we wanted to find,
# and the last element is the Lagrange multiplier, which we can throw away now it has been used.
(x0, y0, λ0) = (-1, -1, 0)
x, y, λ = optimize.root(DL, [x0, y0, λ0]).x
print("x = %g" % x)
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))
print("g(x, y) = %g" % g(x, y))
