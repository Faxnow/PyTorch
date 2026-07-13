import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784, 10)
x = torch.randn(64, 784)
# print(model(x))
# print(model(x).shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

train_dataset = datasets.MNIST(root='.data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='.data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size= input_size, num_classes= num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)
        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device=device)
            target = target.to(device=device)
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            _, predicted = scores.max(1)
            num_correct += (predicted == target).sum()
            num_samples += predicted.size(0)

        print(f'Got {num_correct / num_samples} with accurancy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()
    return num_correct / float(num_samples)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)