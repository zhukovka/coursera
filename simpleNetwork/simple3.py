import numpy as np
# First define our sigma function.
sigma = np.tanh

# z = w1 * a0 + b1
# a1 = sigma(z)
# Next define the feed-forward equation.
def a1 (w1, b1, a0) :
    z = w1 * a0 + b1
    return sigma(z)

# C = (a1 - y)^2
# The individual cost function is the square of the difference between
# the network output and the training data output.
def C (w1, b1, x, y) :
    return (a1(w1, b1, x) - y)**2

# dC   dC   da   dz
# -- = -- * -- * --
# dw   da   dz   dw
# This function returns the derivative of the cost function with
# respect to the weight.
def dCdw (w1, b1, x, y) :
    z = w1 * x + b1
    dCda = 2 * (a1(w1, b1, x) - y) # Derivative of cost with activation
    dadz = 1/np.cosh(z)**2 # derivative of activation with weighted sum z
    dzdw = x # derivative of weighted sum z with weight
    return dCda * dadz * dzdw # Return the chain rule product.

# dC   dC   da   dz
# -- = -- * -- * --
# db   da   dz   db
# This function returns the derivative of the cost function with
# respect to the bias.
# It is very similar to the previous function.
# You should complete this function.
def dCdb (w1, b1, x, y) :
    z = w1 * x + b1
    dCda = 2 * (a1(w1, b1, x) - y)
    dadz = 1/np.cosh(z)**2
    """ Change the next line to give the derivative of
        the weighted sum, z, with respect to the bias, b. """
    dzdb = 1
    return dCda * dadz * dzdb

"""Test your code before submission:"""
# Let's start with an unfit weight and bias.
w1 = 2.3
b1 = -1.2
# We can test on a single data point pair of x and y.
x = 0
y = 1
# Output how the cost would change
# in proportion to a small change in the bias
print( dCdb(w1, b1, x, y) )