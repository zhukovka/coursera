# Import libraries
import numpy as np
from scipy import optimize


# First we define the functions, YOU SHOULD IMPLEMENT THESE
#
def f(x, y):
    return -np.exp(x - y * y + x * y)


# cosh(y)+x−2=0
def g(x, y):
    return np.cosh(y) + x - 2


# Next their derivatives, YOU SHOULD IMPLEMENT THESE
def dfdx(x, y):
    return f(x, y) * (1 + y)


#
def dfdy(x, y):
    return f(x, y) * (-2 * y + x)


#
def dgdx(x, y):
    return 1


#
def dgdy(x, y):
    return np.sinh(y)


#
# Use the definition of DL from previously.
def DL(xyλ):
    [x, y, λ] = xyλ
    return np.array([
        dfdx(x, y) - λ * dgdx(x, y),
        dfdy(x, y) - λ * dgdy(x, y),
        - g(x, y)
    ])


# To score on this question, the code above should set
# the variables x, y, λ, to the values which solve the
# Langrange multiplier problem.

# I.e. use the optimize.root method, as you did previously.
(x0, y0, λ0) = (-1, -1, 0)
x, y, λ = optimize.root(DL, [x0, y0, λ0]).x

print("x = %g" % x)
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))
