import torch
import numpy as np

data = [[1, 2], [3, 4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(np_array)
print(x_np)

x_ones = torch.ones_like(x_np)
x_rand = torch.rand_like(x_np, dtype=torch.float)
print(x_ones)
print(x_rand)