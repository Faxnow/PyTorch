import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                      device=device, requires_grad=True)

print(tensor)
print(tensor.dtype)
print(tensor.device)
print(tensor.shape)
print(tensor.requires_grad)

x = torch.empty(size=(3, 4))
print(x)
x = torch.zeros(size=(3, 4))
print(x)

x = torch.rand((3, 4))
print(x)

x = torch.ones(size=(5, 3))
print(x)

x = torch.eye(5, 5)
print(x)

x = torch.arange(start=2, end=-3, step=-1)
print(x)

x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

x = torch.empty(size=(3, 6)).normal_(mean=0, std=1)
print(x)

x = torch.empty(size=(3, 6)).uniform_(0, 1)
print(x)

x = torch.diag(torch.ones(5))
print(x)

tensor_ = torch.arange(4)
print(tensor_.bool())
print(tensor_.short())
print(tensor_.long())
print(tensor_.half())
print(tensor_.float())
print(tensor_.double())

np_array = np.zeros((3, 3))
tensor__ = torch.from_numpy(np_array)
numpy_array_back = tensor__.numpy()