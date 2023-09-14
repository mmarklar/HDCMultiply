import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import MultiDataset
from network import MultiNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_path = "model.pt"
checkpoint_path = "checkpoint.pt"
train_data_path = "hvs/train_data.pt"

#all_data = MultiDataset()
#train_data, test_data = torch.utils.data.random_split(all_data, [0.2, 0.8])
train_data = torch.load(train_data_path)
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, **kwargs)

# Setup the network and optimizer
net = MultiNet()
net.to(device)
#net = torch.load(model_path)
#optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
epoch = 0

# Load the checkpoint if it exists
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loading checkpoint. Epoch:", epoch)

net.to(device)
net.train()

criterion = nn.MSELoss()
last_loss = 1000
for i in range(10000):
    for data in train_dataloader:
        net_input = data[0].to(device)
        label = data[1].to(device)
        # training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(net_input)
        loss = criterion(output, label)
        last_loss = loss.item()
        loss.backward()
        optimizer.step()
    if i % 1000 == 0:
        print("Train step ", epoch + i, " loss", last_loss)

print("Final loss", last_loss)
# Add the number of times we went through the data
epoch += 10000

#torch.save(net, model_path)
torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    #'loss': loss,
}, checkpoint_path)
