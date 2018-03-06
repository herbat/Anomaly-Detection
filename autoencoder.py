import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F


class EEGData(data.Dataset):

    def __init__(self, set):
        self.set = set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]


# try a fully connected model first, no convnet
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        #encoders
        self.enc1 = nn.Linear(1000, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)
        # decoders
        self.dec1 = nn.Linear(256, 128)
        self.dec2 = nn.Linear(512, 256)
        self.dec3 = nn.Linear(1000, 512)

    def forward(self, x):
        x = F.relu(self.enc3(self.enc2(self.enc1(x)))) # pass thru encoders
        x = F.relu(self.dec1(self.dec2(self.dec3(x)))) # pass thru decoders
        return x

model = AutoEncoder()
crit  = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

tset = pickle.load('trainingset.pkl')
dset = EEGData(tset)
dataloader = data.DataLoader(dset, batch_size=64, shuffle=True)

#TODO: training the network