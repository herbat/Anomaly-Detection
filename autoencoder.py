import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

lr = 1e-3
batch = 64
decay = 1e-5
epochs = 100



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
        self.dec1 = nn.Linear(128, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, 1000)

    def forward(self, x):
        x = F.relu(self.enc3(F.relu(self.enc2(F.relu(self.enc1(x)))))) # pass thru encoders
        x = F.relu(self.dec3(F.relu(self.dec2(F.relu(self.dec1(x)))))) # pass thru decoders
        return x

model = AutoEncoder()
crit  = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = decay)

tset = pickle.load(open('trainingset.pkl', 'rb'))
dset = EEGData(tset)
dataloader = data.DataLoader(dset, batch_size = batch, shuffle = True)

#TODO: training the network
for epoch in range(epochs):
    for data in dataloader:
        eeg = data.float()
        eeg = eeg.view(eeg.size(0), -1)
        eeg = Variable(eeg)
        # ===================forward=====================
        output = model(eeg)
        loss = crit(output, eeg)
        # ===================backward====================
        optim.zero_grad()
        loss.backward()
        optim.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data[0]))

torch.save(model.state_dict(), './eeg_autoencoder.pth')