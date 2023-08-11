import torch
import numpy as np

t = torch.rand(3, 3)

if torch.cuda.is_available():
    tensor = t.to('cuda')

n = t.numpy()
t.add_(1)
print(n)

# print(tensor)

# agg = tensor.sum()
# agg_item = agg.item()
# print(agg_item, type(agg_item))