import numpy as np
import matplotlib.pyplot as plt

# feed-forward equations,
# a_n = σ(z_n)
#
# z_n = W_n*a_n_1+b_n

# In this worksheet we will use the logistic function as our activation function, rather than the more familiar  tanhtanh .
#
#           1
# σ(z)= --------
#       1+exp(−z)
#
# Here is the activation function and its derivative.
sigma = lambda z: 1 / (1 + np.exp(-z))
d_sigma = lambda z: np.cosh(z / 2) ** (-2) / 4


# This function initialises the network with it's structure, it also resets any training already done.
def reset_network(n1=6, n2=7, random=np.random):
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2


# This function feeds forward each activation to the next layer. It returns all weighted sums and activations.
def network_function(a0):
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3


# This is the cost function of a neural network with respect to a training set.
def cost(x, y):
    return np.linalg.norm(network_function(x)[-1] - y) ** 2 / x.size


#                   ---   ②
#       ---   ①
#                   ---   ②
#       ---   ①
#                   ---   ②
#       ---   ①                 ---   ③
#                   ---   ②
# ⓪                             ---   ③
#                   ---   ②
#       ---   ①
#                   ---   ②
#       ---   ①
#                   ---   ②
#       ---   ①
# a(0)  W(1)  a(1)  W(2)  a(2)  W(2)  a(3)
#
# a(0)                   - [1x1]
# W(1)                   - [6x1]
# a(1) = σ(W1 @ a0 + b1) - [6x1]
# W(2)                   - [7x6]
# a(2) = σ(W2 @ a1 + b2) - [7x1]
# W(3)                   - [2x7]
# a(3)                   - [2x1]

#
# Jacobians as,
#          ∂C
# JW(3)= ------
#         ∂W(3)
#
#          ∂C
# Jb(3)= -------
#         ∂b(3)
#
# etc., where  C  is the average cost function over the training set. i.e.,
#     1
# C = - ∑ Ck
#     N k
#
#   ∂C      ∂C    ∂a(3)   ∂z(3)
# ----- = ----- * ----- * -----
# ∂W(3)   ∂a(3)   ∂z(3)   ∂W(3)
#
#
#  ∂C     ∂C    ∂a(3)  ∂z(3)
# ---- = ---- * ---- * -----
# ∂b(3)  ∂a(3)  ∂z(3)  ∂b(3)
#
#
#   ∂C
# ----- = 2(a(3)−y)
# ∂a(3)
#
# ∂a(3)
# ----- = σ′(z(3))
# ∂z(3)
#
# ∂z(3)
# ----- = a(2)
# ∂W(3)
#
# ∂z(3)
# ----- = 1
# ∂b(3)
#

# Jacobian for the third layer weights.
#   ∂C      ∂C    ∂a(3)   ∂z(3)
# ----- = ----- * ----- * -----
# ∂W(3)   ∂a(3)   ∂z(3)   ∂W(3)
def J_W3(x, y):
    # First get all the activations and weighted sums at each layer of the network.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # We'll use the variable J to store parts of our result as we go along, updating it in each line.
    #  ∂C      ∂C        ∂a(3)      ∂z(3)
    # ----- = ----- 🅰️ * ----- 🅱️ * ----- 🅾️
    # ∂W(3)   ∂a(3)      ∂z(3)      ∂W(3)
    # Firstly, we calculate dC/da3, using the expressions above.
    #   ∂C
    # ----- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)  # - [2x1]
    # Next multiply the result we've calculated by the derivative of sigma, evaluated at z3.
    # ∂a(3)
    # ----- = σ′(z(3)) 🅱️
    # ∂z(3)
    J = J * d_sigma(z3)  # 🅰️ * 🅱️ - [2x1]
    # Then we take the dot product (along the axis that holds the training examples) with the final partial derivative,
    # i.e. dz3/dW3 = a2

    # ∂z(3)
    # ----- = a(2) 🅾️
    # ∂W(3)

    # and divide by the number of training examples, for the average over all training examples.
    #   a(2)[7x1] * J[2x1] ❌
    # a(2).T[1x7] * J[2x1] ✅ - [1x1]
    J = J @ a2.T / x.size  # (🅰️ * 🅱️) * 🅾️ / N
    # Finally return the result out of the function.
    return J


# In this function, you will implement the jacobian for the bias.
# As you will see from the partial derivatives, only the last partial derivative is different.
# The first two partial derivatives are the same as previously.
#  ∂C     ∂C    ∂a(3)  ∂z(3)
# ---- = ---- * ---- * -----
# ∂b(3)  ∂a(3)  ∂z(3)  ∂b(3)
def J_b3(x, y):
    # As last time, we'll first set up the activations.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Next you should implement the first two partial derivatives of the Jacobian.
    # ===COPY TWO LINES FROM THE PREVIOUS FUNCTION TO SET UP THE FIRST TWO JACOBIAN TERMS===
    #  ∂C
    # ---- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)
    # Next multiply the result we've calculated by the derivative of sigma, evaluated at z3.
    # ∂a(3)
    # ----- = σ′(z(3)) 🅱️
    # ∂z(3)
    J = J * d_sigma(z3)  # 🅰️ * 🅱️
    # For the final line, we don't need to multiply by dz3/db3, because that is multiplying by 1.
    # ∂z(3)
    # ----- = 1
    # ∂b(3)
    # We still need to sum over all training examples however.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


# Jacobian for the Layer 2. The partial derivatives for this are,
#   ∂C     ∂C        ∂a(3)      ∂a(2)      ∂z(2)
# ----- = ----- 🅰️ * ----- 🅱️ * ----- 🅾️ * ----- 🆑
# ∂W(2)   ∂a(3)      ∂z(2)      ∂z(2)      ∂W(2)
#
#
#   ∂C     ∂C        ∂a(3)      ∂a(2)      ∂z(2)
# ----- = ----- 🅰️ * ----- 🅱️ * ----- 🅾️ * ----- 🆑
# ∂b(2)   ∂a(3)      ∂z(2)      ∂z(2)      ∂b(2)
#
#
# ∂a(3)   ∂a(3)   ∂z(3)
# ----- = ----- * ----- = σ′(z(3))W(3) 🅱️
# ∂a(2)   ∂z(3)   ∂a(2)
#

# Compare this function to J_W3 to see how it changes.
# There is no need to edit this function.
def J_W2(x, y):
    # The first two lines are identical to in J_W3.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    #   ∂C
    # ----- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    # ∂a(3)   ∂a(3)          ∂z(3)
    # ----- = -----  🅱️[1] * ----- 🅱️[2] = σ′(z(3))W(3) 🅱️
    # ∂z(2)   ∂z(3)          ∂a(2)
    J = J * d_sigma(z3)  # (🅰️ * 🅱️[1])
    J = (J.T @ W3).T  # (🅰️ * 🅱️[1]) * 🅱️[2]
    # then the final lines are the same as in J_W3 but with the layer number bumped down.
    # ∂a(2)
    # ----- = σ′(z(2)) 🅾️
    # ∂z(2)
    J = J * d_sigma(z2)  # (🅰️ * 🅱️) * 🅾️
    # ∂z(2)
    # ----- = a(1) 🆑
    # ∂W(2)
    J = J @ a1.T / x.size  # (🅰️ * 🅱️) * 🅾️ *  🆑 / N
    return J


# As previously, fill in all the incomplete lines.
# ===YOU SHOULD EDIT THIS FUNCTION===
def J_b2(x, y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    #   ∂C
    # ----- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    # ∂a(3)   ∂a(3)          ∂z(3)
    # ----- = -----  🅱️[1] * ----- 🅱️[2] = σ′(z(3))W(3) 🅱️
    # ∂a(2)   ∂z(3)          ∂a(2)
    J = J * d_sigma(z3)  # (🅰️ * 🅱️[1])
    J = (J.T @ W3).T  # (🅰️ * 🅱️[1]) * 🅱️[2]
    # ∂a(2)
    # ----- = σ′(z(2)) 🅾️
    # ∂z(2)
    J = J * d_sigma(z2)  # (🅰️ * 🅱️) * 🅾️
    # ∂z(2)
    # ----- = 1 🆑
    # ∂b(2)
    # We still need to sum over all training examples however.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


# Jacobian for the Layer 2. The partial derivatives for this are,
#   ∂C     ∂C        ∂a(3)      ∂a(2)      ∂a(1)      ∂z(1)
# ----- = ----- 🅰️ * ----- 🅱️ * ----- 🅱️ * ----- 🅾️ * ----- 🆑
# ∂W(1)   ∂a(3)      ∂a(2)      ∂a(1)      ∂z(1)      ∂W(1)
#
#
#   ∂C     ∂C        ∂a(3)      ∂a(2)      ∂a(1)      ∂z(1)
# ----- = ----- 🅰️ * ----- 🅱️ * ----- 🅱️ * ----- 🅾️ * ----- 🆑
# ∂b(1)   ∂a(3)      ∂a(2)      ∂a(1)      ∂z(1)      ∂b(1)
#
#
# ∂a(3)   ∂a(3)   ∂z(3)
# ----- = ----- * ----- = σ′(z(3))W(3) 🅱️
# ∂a(2)   ∂z(3)   ∂a(2)
#
# ∂a(2)   ∂a(2)   ∂z(2)
# ----- = ----- * ----- = σ′(z(2))W(2) 🅱️
# ∂a(1)   ∂z(2)   ∂a(1)
#
# Fill in all incomplete lines.
# ===YOU SHOULD EDIT THIS FUNCTION===
def J_W1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    #   ∂C
    # ----- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    # ∂a(3)   ∂a(3)          ∂z(3)
    # ----- = -----  🅱️[1] * ----- 🅱️[2] = σ′(z(3))W(3) 🅱️
    # ∂a(2)   ∂z(3)          ∂a(2)
    J = J * d_sigma(z3)  # (🅰️ * 🅱️[1])
    J = (J.T @ W3).T  # (🅰️ * 🅱️[1]) * 🅱️[2]
    # ∂a(2)   ∂a(2)   ∂z(2)
    # ----- = ----- * ----- = σ′(z(2))W(2) 🅱️
    # ∂a(1)   ∂z(2)   ∂a(1)
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    # ∂a(1)
    # ----- = σ′(z(1)) 🅾️
    # ∂z(1)
    J = J * d_sigma(z1)  # (🅰️ * 🅱️) * 🅾️
    # ∂z(1)
    # ----- = a(0) 🆑
    # ∂W(1)
    J = J @ a0.T / x.size
    return J

# Fill in all incomplete lines.
# ===YOU SHOULD EDIT THIS FUNCTION===
def J_b1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    #  ∂C
    # ----- = 2(a(3)−y) 🅰️
    # ∂a(3)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    # ∂a(3)   ∂a(3)          ∂z(3)
    # ----- = -----  🅱️[1] * ----- 🅱️[2] = σ′(z(3))W(3) 🅱️
    # ∂a(2)   ∂z(3)          ∂a(2)
    J = J * d_sigma(z3)  # (🅰️ * 🅱️[1])
    J = (J.T @ W3).T  # (🅰️ * 🅱️[1]) * 🅱️[2]
    # ∂a(2)   ∂a(2)   ∂z(2)
    # ----- = ----- * ----- = σ′(z(2))W(2) 🅱️
    # ∂a(1)   ∂z(2)   ∂a(1)
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    # ∂a(1)
    # ----- = σ′(z(1)) 🅾️
    # ∂z(1)
    J = J * d_sigma(z1)  # (🅰️ * 🅱️) * 🅾️
    # ∂z(1)
    # ----- = 1 🆑
    # ∂b(1)
    # We still need to sum over all training examples however.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J