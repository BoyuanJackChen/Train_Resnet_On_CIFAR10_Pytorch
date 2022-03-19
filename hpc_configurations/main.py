import os
print("Hello from the compute node", os.uname()[1])

import torch
print(torch.tensor([[1., -1.], [1., -1.]]))
