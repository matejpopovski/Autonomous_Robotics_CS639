import numpy as np
from matplotlib import pyplot as plt

points = np.load('points.npy')

plt.scatter(points[:, 0], points[:, 1])
plt.xlim([-2, 2])
plt.ylim([-3, 3])
plt.show()
