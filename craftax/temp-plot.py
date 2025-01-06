# %%
import numpy as np

X = [(1, np.array(0.0050531, dtype=np.float32), np.array(0.00502932, dtype=np.float32)),
    (2, np.array(0.00647349, dtype=np.float32), np.array(0.00660622, dtype=np.float32)),
    (3, np.array(0.00718547, dtype=np.float32), np.array(0.00711286, dtype=np.float32)),
    (4, np.array(0.00857076, dtype=np.float32), np.array(0.00851011, dtype=np.float32)),
    (5, np.array(0.00996101, dtype=np.float32), np.array(0.00995684, dtype=np.float32)),
    (6, np.array(0.01073824, dtype=np.float32), np.array(0.0106982, dtype=np.float32)),
    (7, np.array(0.01221034, dtype=np.float32), np.array(0.01219499, dtype=np.float32)),
    (8, np.array(0.01336098, dtype=np.float32), np.array(0.01336718, dtype=np.float32)),
    (9, np.array(0.01450219, dtype=np.float32), np.array(0.01448643, dtype=np.float32)),
    (10, np.array(0.01557298, dtype=np.float32), np.array(0.01556277, dtype=np.float32)),
    (11, np.array(0.01688518, dtype=np.float32), np.array(0.01687217, dtype=np.float32)),
    (12, np.array(0.01806343, dtype=np.float32), np.array(0.01804876, dtype=np.float32)),
    (13, np.array(0.01913111, dtype=np.float32), np.array(0.01896584, dtype=np.float32)),
    (14, np.array(0.02058573, dtype=np.float32), np.array(0.02039671, dtype=np.float32)),
    (15, np.array(0.02144882, dtype=np.float32), np.array(0.02133465, dtype=np.float32)),
    (16, np.array(0.02249441, dtype=np.float32), np.array(0.02250171, dtype=np.float32))]

indices = [item[0] for item in X]
min_values = [min(item[1], item[2])*1000 for item in X]

print(indices)
print(min_values)

baseline = 0.005911*1000

# %%
import matplotlib.pyplot as plt

plt.plot(indices, min_values, label='Multi-agent Implementation')
plt.axhline(y=baseline, color='r', linestyle='--', label='Original Implementation')

plt.xlabel('Number of Agents')
plt.ylabel('Average Runtime (ms)')
plt.title('craftax_step()')
plt.legend()
plt.show()

# %%
