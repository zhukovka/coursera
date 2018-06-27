# Import libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

points = np.array([[0.4, 0.1], [0.5, 0.25], [0.6, 0.55], [0.7, 0.75], [0.8, 0.85]])

xs = points[:, 0]
ys = points[:, 1]


#             _
#     ∑(x_i - x)y_i          _    _
# m = --------_----      c = y - mx
#     ∑(x_i - x)^2

# Here the function is defined
def linfit(xdat, ydat):
    # Here xbar and ybar are calculated
    xbar = np.sum(xdat) / len(xdat)
    ybar = np.sum(ydat) / len(ydat)

    # Insert calculation of m and c here. If nothing is here the data will be plotted with no linear fit
    sum1 = np.sum((xdat - xbar) * ydat)
    sum2 = np.sum((xdat - xbar) ** 2)
    m = sum1 / sum2
    c = ybar - m * xbar
    # Return your values as [m, c]
    return [m, c]


print(linfit(xs, ys))

# Use the stats.linregress() method to evaluate regression
regression = stats.linregress(xs, ys)


# plt.plot(slope, intercept, regression)
# plt.show()
