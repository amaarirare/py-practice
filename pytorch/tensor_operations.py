import torch
import numpy as np

tensor = torch.rand(3, 3)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# n = t.numpy()
# t.add_(1)
# print(n)

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(t)
print(n)

# print(tensor)

# agg = tensor.sum()
# agg_item = agg.item()
# print(agg_item, type(agg_item))