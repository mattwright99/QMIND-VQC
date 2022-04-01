
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F


from model import QuanvNet, QuanvLayer


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

im_size = 8
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((im_size,im_size)),
                    ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((im_size,im_size)),
                    ])),
    batch_size=batch_size_test, shuffle=True)


class CustomQunvNet(nn.Module):
    """Layout of Quanv model that can be modified on the fly"""
    def __init__(self, input_size=8, shots=128):
        super(CustomQunvNet, self).__init__()

        self.fc_size = (input_size - 3)**2 * 16  # output size of convloving layers
        self.quanv = QuanvLayer(in_channels=1, out_channels=2, kernel_size=2, shots=shots)
        self.conv = nn.Conv2d(2, 16, kernel_size=3)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # this is where we build our entire network
        # whatever layers of quanvolution, pooling,
        # convolution, dropout, flattening,
        # fully connectecd layers, go here
        x = F.relu(self.quanv(x))
        x = F.relu(self.conv(x))
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Model trianing
model = QuanvNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

epochs = 5
train_losses = []
train_accs = []

model.train()
for epoch in range(epochs):
    epoch_train_acc = []  # sums training accuracy for a given epoch
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)
        pred = output.argmax(axis=1)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.item())

        acc = (pred == target).float().sum()/ len(target)
        epoch_train_acc.append(acc.item())

    train_losses.append(sum(epoch_loss) / len(epoch_loss))
    train_accs.append(sum(epoch_train_acc) / len(epoch_train_acc))
    print('Training [{:.0f}%] \t Loss: {:.4f} \t Accuracy: {:.4}'.format(
        100. * (epoch + 1) / epochs, train_losses[-1]), train_accs[-1])


## TODO: visualize results and training
fig = plt.figure()
plt.plot(range(len(train_losses)), train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig("training.pdf", dpi=800)
plt.show()

np.save('train_loss', train_losses)
np.save('train_acc', train_accs)
