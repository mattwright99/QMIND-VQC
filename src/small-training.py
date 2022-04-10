
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F


from model import QuanvNet, QuanvLayer


n_epochs = 10

n_train_samples = 512
n_test_samples = 512
batch_size_train = 32
batch_size_test = 512
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 123
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

im_size = 8

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


class CustomQunvNet(nn.Module):
# # ---------------OPTION 1: SINGLE CONV LAYER --------------------------------
#     """Layout of Quanv model that can be modified on the fly"""
#     def __init__(self, input_size=8, shots=128):
#         super(CustomQunvNet, self).__init__()

#         # self.fc_size = (input_size - 3)**2 * 16  # output size of convloving layers
#         # # this ^ is 400
#         self.fc_size = 6 * 6 * 16 # = 576
#         # self.quanv = QuanvLayer(in_channels=1, out_channels=2, kernel_size=2, shots=shots)
#         self.conv = nn.Conv2d(1, 16, kernel_size=3)
#         # self.dropout = nn.Dropout2d()
#         self.fc1 = nn.Linear(self.fc_size, 64)
#         self.fc2 = nn.Linear(64, 2)

        

#     def forward(self, x):
#         # this is where we build our entire network
#         # whatever layers of quanvolution, pooling,
#         # convolution, dropout, flattening,
#         # fully connectecd layers, go here
#         # x = F.relu(self.quanv(x))
#         x = F.relu(self.conv(x))
#         # x = x.view(-1, self.fc_size)
#         x = torch.flatten(x, start_dim=1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

#     #--------------------------------------------------------------------------



    # ---------------OPTION 2: REPLACE QUANV WITH CONV --------------------------------
    """Layout of Quanv model that can be modified on the fly"""
    def __init__(self, input_size=8, shots=128):
        super(CustomQunvNet, self).__init__()

        # self.fc_size = (input_size - 3)**2 * 16 # = 400  # output size of convloving layers
        # # this ^ is 400
        # self.fc_size = 6 * 6 * 16 # = 576
        self.fc_size = 4 * 4 * 32 # = 512
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 2)

        

    def forward(self, x):
        # this is where we build our entire network
        # whatever layers of quanvolution, pooling,
        # convolution, dropout, flattening,
        # fully connectecd layers, go here
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#-------------------------------------------------------------------------------------

# Model trianing
model = CustomQunvNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

test_data, test_target = next(iter(test_loader))  # just access first batch

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

        output = model(data)
        pred = output.argmax(axis=1)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss.append(loss.item())

        acc = (pred == target).float().sum()/ len(target)
        epoch_train_acc.append(acc.item())

        # testing acc
        test_pred = model(test_data).argmax(axis=1)
        test_acc = (test_pred == test_target).float().sum()/ len(test_target)
        epoch_test_acc.append(test_acc.item())
    
    torch.save(model.state_dict(), f'model_dict_e{epoch}')
    
    train_losses.append(sum(epoch_train_loss) / len(epoch_train_loss))
    train_accs.append(sum(epoch_train_acc) / len(epoch_train_acc))
    test_accs.append(sum(epoch_test_acc) / len(epoch_test_acc))

    print('Training Progress: [{:.0f}%] \t Loss: {:.4f} \t Train Acc: {:.4} \t Test Acc: {:.4}'.format(
        100. * (epoch + 1) / n_epochs, train_losses[-1], train_accs[-1], test_accs[-1]))


np.save('train_loss', train_losses)
np.save('train_accs', train_accs)
np.save('test_accs', test_accs)

## TODO: visualize results and training
fig = plt.figure()
plt.plot(range(len(train_losses)), train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig("training.pdf", dpi=800)
# plt.show()

