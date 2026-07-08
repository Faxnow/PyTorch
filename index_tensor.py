import torch

batch_size = 10
features = 24
x = torch.rand((batch_size, features))

print(x[0].shape)
print(x[:, 0])
print(x)
print(x[2, 0:10])
x[0, 0] = 100

x = torch.arange(10)
indices = [4, 6, 2]
print(x[indices])

x = torch.rand((4, 6))
rows = torch.tensor([2, 1])
cols = torch.tensor([4, 3])
print(x[rows, cols].shape)

x = torch.arange(10)
print(x[(x < 2) | (x > 9)])
print(x[x.remainder(2) == 0])


print(torch.where(x > 5, x, x*2))
print(torch.tensor([1, 3, 2, 3, 0, 0, 3, 1, 1]).unique())
print(x.ndimension())
print(x.numel())
