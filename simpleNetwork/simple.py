import numpy as np
σ = np.tanh
w1 = 1.3
b1 = -0.1


# Let's assume we want to train the network to give a NOT function, that is if you input 1 it returns 0, and if you input 0 it returns 1.
#
# For simplicity, let's use, \sigma(z) = \tanh(z)σ(z)=tanh(z), for our activation function, and randomly initialise our weight and bias to w^{(1)}=1.3w
# (1)
#  =1.3 and b^{(1)} = -0.1b
# (1)
#  =−0.1.
#
# Use the code block below to see what output values the neural network initially returns for training data.

# Then we define the neuron activation.
def a1(a0) :
    return σ(w1 * a0 + b1)

# Finally let's try the network out!
# Replace x with 0 or 1 below,
a1(0)

# First set up the network.
sigma = np.tanh
W = np.array([[-2, 4, -1],[6, 0, -3]])
b = np.array([0.1, -2.5])

# Define our input vector
x = np.array([0.3, 0.4, 0.1])
x1 = sigma(W @ x + b)
print(x1)
# Calculate the values by hand,
# and replace a1_0 and a1_1 here (to 2 decimal places)
# (Or if you feel adventurous, find the values with code!)
# a1 = np.array(x1)