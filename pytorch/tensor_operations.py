import torch
import numpy as np

tensor = torch.ones(4, 4)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
