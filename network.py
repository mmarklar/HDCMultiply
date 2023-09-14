import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        self.fc1 = nn.Linear(200, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = MultiNet()
    total_params = sum(
            param.numel() for param in net.parameters()
    )
    print(net, total_params)
