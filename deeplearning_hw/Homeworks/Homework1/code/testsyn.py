import numpy as np

x1 = np.arange(4.0).reshape((2, 2))
x2 = np.arange(2.0)
y= np.multiply(x1, x2)
print("x1 = {}, x2 = {}".format(x1, x2))
print("x1*x2 = {}".format(y+2))
print(y.shape[0])