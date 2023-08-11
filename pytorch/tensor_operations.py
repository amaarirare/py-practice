import torch
import numpy as np

tensor = torch.rand(3, 3)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# y0 = tensor 
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# z1 = tensor * tensor

# print(y0)
# print(y1)
# print(y2)

# print(z1)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

