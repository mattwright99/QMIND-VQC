
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

from model import QuanvNet, QuanvLayer

# plotting stuff
plt.rcParams['font.size'] = 18
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1


n_epochs = 2

# Feel free to set these to increase or decrease the size of the training 
im_size = 4  # side length used to scale loaded images
n_train_samples = 8
n_test_samples = 16
batch_size_train = 8
batch_size_test = 16

learning_rate = 0.01
momentum = 0.5

random_seed = 123
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load training data
X_train = datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize((im_size,im_size)),
    ])
)
# Select only labels 0 and 1
idx = np.append(np.where(X_train.targets == 0)[0][:n_train_samples], 
                np.where(X_train.targets == 1)[0][:n_train_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size_train, shuffle=True)

# Load testing data
X_test = datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize((im_size,im_size)),
    ])
)
idx = np.append(np.where(X_test.targets == 0)[0][:n_test_samples], 
                np.where(X_test.targets == 1)[0][:n_test_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size_test, shuffle=True)


class CustomQuanvNet(nn.Module):
    """Layout of Quanv model that can be modified on the fly"""
    def __init__(self, input_size=im_size, shots=256, n_classes=2):
        super(CustomQuanvNet, self).__init__()

        self.fc_size = (input_size - 3)**2 * 16  # output size of convloving layers
        self.quanv = QuanvLayer(in_channels=1, out_channels=2, kernel_size=2, shots=shots)
        self.conv = nn.Conv2d(2, 16, kernel_size=3)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_size, 32)
        self.fc2 = nn.Linear(32, n_classes)

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
model = CustomQuanvNet(
    input_size=im_size, shots=256, n_classes=2
)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

test_data, test_target = next(iter(test_loader))  # just access first batch

print('Training...')
model.train()
for epoch in range(n_epochs):
    # keep track of accuracy and loss for a given epoch
    epoch_train_acc = []
    epoch_train_loss = []
    # TODO: validation score
    epoch_test_acc = []
    epoch_test_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Compute batch predictions
        output = model(data)
        pred = output.argmax(axis=1)
        
        # Compute loss and backprop
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        # Keep track of training progress over batchs
        epoch_train_loss.append(loss.item())
        # training acc
        acc = (pred == target).float().sum()/ len(target)
        epoch_train_acc.append(acc.item())
        # testing acc
        test_pred = model(test_data).argmax(axis=1)
        test_acc = (test_pred == test_target).float().sum()/ len(test_target)
        epoch_test_acc.append(test_acc.item())
        print(f'Batch {batch_idx} acc: {test_acc.item()}')
    
    # Save model state for each epoch
    torch.save(model.state_dict(), f'model_dict_e{epoch}')
    
    # Keep track of training progress over epochs
    train_losses.append(sum(epoch_train_loss) / len(epoch_train_loss))
    train_accs.append(sum(epoch_train_acc) / len(epoch_train_acc))
    test_accs.append(sum(epoch_test_acc) / len(epoch_test_acc))

    print('Training [{:.0f}%] \t Loss: {:.4f} \t Accuracy: {:.4} \t Test Accuracy: {:.4}'.format(
        100. * (epoch + 1) / n_epochs, train_losses[-1], train_accs[-1], test_accs[-1]))


np.save('train_loss', train_losses)
np.save('train_accs', train_accs)
np.save('test_accs', test_accs)

# Visualize results and training
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.2, .2, .6, .6])

ax.plot(range(n_epochs), train_accs, 'b-', label='Train')
ax.plot(range(n_epochs), test_accs, 'r-', label='Test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.grid(True, linestyle=':')
ax.legend()

plt.savefig(f'training_acc.pdf', dpi=800, bbox_inches="tight")
plt.show()

