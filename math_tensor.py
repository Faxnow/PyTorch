import torch

x = torch.tensor([1, 6, 3])
y = torch.tensor([4, 6, 2])

z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
z = x + y

z = x - y
print(z)

z = torch.true_divide(x, y)
print(z)

t = torch.zeros(3)
t.add_(x)
t += x
print(t)

z = x.pow(2)
z = x ** 2
print(z)

z = x  > 0
print(z)
z = x < 0
print(z)

x1 = torch.rand((4, 3))
x2 = torch.rand((3, 6))
x3 = torch.mm(x1, x2)
print(x3)
x3 = x1.mm(x2)
print(x3)

exp = torch.rand(5, 5)
print(exp.matrix_power(4))

z = x * y
print(z)

z = torch.dot(x, y)
print(z)

batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)

x1 = torch.rand((3, 3))
x2 = torch.rand((1, 3))

z = x1 - x2
print(z)
z = x1 ** x2
print(z)

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=2, max=7)

x = torch.tensor([1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)
print(z)