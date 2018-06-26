import numpy as np

# a1=σ(z1)
#
# z1 = w1 * a0 + b1
#
# Where we've introduced z1 as the weighted sum of activation and bias.
#
# We can formalise how good (or bad) our neural network is at getting the desired behaviour.
# For a particular input, x, and desired output y, we can define the cost of that specific training example as the square of the difference between the network's output and the desired output, that is,
#
# Ck = (a1 - y)^2
#
σ = np.tanh
