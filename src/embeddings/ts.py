import numpy as np
m = np.array([1, 2])
n = np.array([3, 4])
a = np.array([[1, 2], [3, 4], [1, 1]])
c = np.array([[2, 3], [8, 9]])
b = np.vstack([a, c])
print(b)