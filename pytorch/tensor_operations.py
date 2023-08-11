import torch
import numpy as np

tensor = torch.rand(3, 3)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

print(tensor)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))