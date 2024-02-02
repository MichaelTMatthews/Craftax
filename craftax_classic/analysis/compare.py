import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("craftax.csv")
y = np.loadtxt("crafter.csv")

x = x.reshape((9 * 7, 9 * 7, 3)).astype(int)
y = y.reshape((9 * 7, 9 * 7, 3)).astype(int).transpose((1, 0, 2))

z = np.concatenate((x, y), axis=1)

plt.imshow(z)
plt.show()
