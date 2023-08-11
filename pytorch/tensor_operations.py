import torch
import numpy as np

tensor = torch.rand(3, 3)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
