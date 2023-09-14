import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device for dataset is: ", device)
class MultiDataset(Dataset):
    def __init__(self): 
        self.inputs = torch.load("hvs/equations.pt")#.to(device)
        self.labels = torch.load("hvs/solutions.pt")#.to(device)
        #inputs = torch.load("hvs/equations.pt")
        #labels = torch.load("hvs/solutions.pt")

        #print("Inputs shape", inputs.shape, "label shape", labels.shape)

        #self.data = torch.stack((inputs, labels), -1).to(device)
        #print("Stacked shape", self.data.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        #return self.data[idx]
        return self.inputs[idx], self.labels[idx]
