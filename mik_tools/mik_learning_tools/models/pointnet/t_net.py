import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Tnet(nn.Module):
   def __init__(self, k=3):
      super(Tnet, self).__init__()
      self.k = k
      self.conv1 = nn.Conv1d(self.k, 64, 1)
      self.conv2 = nn.Conv1d(64, 128, 1)
      self.conv3 = nn.Conv1d(128, 1024, 1)
      self.fc1 = nn.Linear(1024, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, self.k*self.k)
      self.relu = nn.ReLU()

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)

   def forward(self, input):
      # Input shape: (B, k, N)
      bs = input.size(0)
      mod = F.relu(self.bn1(self.conv1(input)))
      mod = F.relu(self.bn2(self.conv2(mod)))
      mod = F.relu(self.bn3(self.conv3(mod)))
      pool = nn.MaxPool1d(mod.size(-1))(mod)
      flat = nn.Flatten(1)(pool)
      mod = F.relu(self.bn4(self.fc1(flat)))
      mod = F.relu(self.bn5(self.fc2(mod)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
      if mod.is_cuda:
        init = init.cuda()
      matrix = self.fc3(mod).view(-1, self.k, self.k) + init
      return matrix


class TnetSymmetric(nn.Module):
   def __init__(self, k=3):
      super(TnetSymmetric, self).__init__()
      self.k = k
      self.conv1 = nn.Conv1d(self.k, 64, 1)
      self.conv2 = nn.Conv1d(64, 128, 1)
      self.conv3 = nn.Conv1d(128, 1024, 1)
      self.fc1 = nn.Linear(1024, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, self.k * self.k*2)
      self.relu = nn.ReLU()

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)

   def forward(self, input):
      # Input shape: (B, k, N)
      bs = input.size(0)
      mod = F.relu(self.bn1(self.conv1(input)))
      mod = F.relu(self.bn2(self.conv2(mod)))
      mod = F.relu(self.bn3(self.conv3(mod)))
      pool = nn.MaxPool1d(mod.size(-1))(mod)
      flat = nn.Flatten(1)(pool)
      mod = F.relu(self.bn4(self.fc1(flat)))
      mod = F.relu(self.bn5(self.fc2(mod)))
      mod = self.fc3(mod)
      mod_1, mod_2 = torch.split(mod, [self.k*self.k, self.k*self.k], dim=-1)
      matrix_1 = mod_1.reshape(-1, self.k, self.k)
      matrix_2 = mod_2.reshape(-1, self.k, self.k)
      matrix = torch.einsum('bij,bkj->bik', matrix_1, matrix_1) + torch.einsum('bij,bkj->bik', matrix_2, matrix_2)
      return matrix

