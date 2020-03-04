from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F


class DatasetAutoEncoder(Dataset):


    def __init__(self, csv_folder:str, split='train', trans: callable = None,norm=False):
        'Initialization'
        self.csv_folder = csv_folder
        if norm:
          scale_train = StandardScaler()
          if split == 'validation':
            tmp = pd.read_hdf(csv_folder+'/train.hdf5')
            self.list_sig = pd.read_hdf(csv_folder+'/'+split+'.hdf5')
            self.list_sig = pd.DataFrame(scale_train.fit(tmp).transform(self.list_sig))
          else:
            self.list_sig = pd.read_hdf(csv_folder+'/'+split+'.hdf5')
            self.list_sig = pd.DataFrame(scale_train.fit(self.list_sig).transform(self.list_sig))
        else:
          self.list_sig = pd.read_hdf(csv_folder+'/'+split+'.hdf5')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_sig)


    def __getitem__(self, index):
        'Generates one sample of data'
        sig = torch.tensor(self.list_sig.iloc[index].values, dtype=torch.float32)
        sig = sig.view(1,61440)
        return sig




import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DownSample,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7,padding=3)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=7,padding=3)
        self.pool  = nn.AvgPool1d(kernel_size = 8,stride = 8)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        return x

class Encoder(nn.Module):

    def __init__(self, latent_dim : int = 128):
        super(Encoder,self).__init__()
        #len = 61440

        self.down1 = DownSample(in_channels=1, out_channels= 4)
        self.down2 = DownSample(in_channels=4, out_channels= 8)
        self.down3 = DownSample(in_channels=8, out_channels= 16)
        self.linear1 = nn.Linear(in_features=16*120, out_features=latent_dim)
        self.flatten = nn.Flatten()

    def forward(self,x):

      x = self.down1(x)
      x = self.down2(x)
      x = self.down3(x)
      x = self.flatten(x)
      x = self.linear1(x)
      return x



class AutoEncoder(nn.Module):
    def __init__(self, latent_dim : int = 128):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(latent_dim = latent_dim)
        self.decoder = Decoder(latent_dim = latent_dim)

    def forward(self,x):
        latent_vector = self.encoder(x)
        x = F.relu(x)
        output = self.decoder(latent_vector)

        return output
