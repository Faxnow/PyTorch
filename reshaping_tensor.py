import torch

x = torch.arange(9)

x_matrix = x.view(3, 3)
print(x_matrix)
print(x_matrix.shape)
x_matrix = x.reshape(3, 3)
print(x_matrix)

y = x_matrix.t()
print(y)
print(y.contiguous().view(9))


x1 = torch.randn((3, 6))
x2 = torch.randn((3, 6))
print(torch.cat((x1, x2), dim=1))
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)
print(z)
print(z.shape)

batch = 64
x = torch.rand((batch, 3, 6))
z = x.view(batch, -1)
print(z)
print(z.shape)

z = x.permute(0, 2, 1)
print(z)
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0))
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1))
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x)
print(x.shape)