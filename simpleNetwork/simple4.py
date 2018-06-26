import numpy as np
sigma = np.tanh

# Let's use a random initial weight and bias.
W = np.array([[-0.94529712, -0.2667356 , -0.91219181],
              [ 2.05529992,  1.21797092,  0.22914497]])
b = np.array([ 0.61273249,  1.6422662 ])

# z = W @ a0 + b
# a1 = sigma(z)
# define our feed forward function
def a1 (a0) :
    # Notice the next line is almost the same as previously,
    # except we are using matrix multiplication rather than scalar multiplication
    # hence the '@' operator, and not the '*' operator.
    z = W @ a0 + b
    # Everything else is the same though,
    return sigma(z)

# Next, if a training example is,
x = np.array([0.1, 0.5, 0.6])
y = np.array([0.25, 0.75])


# Then the cost function is,
d = a1(x) - y # Vector difference between observed and expected activation
C = d @ d # Absolute value squared of the difference.