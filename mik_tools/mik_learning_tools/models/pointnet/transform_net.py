import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mik_tools.mik_learning_tools.models.pointnet.t_net import Tnet, TnetSymmetric


class Transform(nn.Module):
    def __init__(self, num_in_features=3):
        super().__init__()
        self.num_in_features = num_in_features
        self.input_transform = self._get_transform_network(k=self.num_in_features)
        self.feature_transform = self._get_transform_network(k=64)

        self.conv1 = nn.Conv1d(self.num_in_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def _get_transform_network(self, k):
        return Tnet(k=k)

    def forward(self, input):
        # input shape (B, N, 3)
        input_t = input.transpose(-1, -2)  # (B, 3, N)
        matrix3x3 = self.input_transform(input_t)  # (B, 3, 3)
        # batch matrix multiplication
        mod = torch.bmm(input, matrix3x3).transpose(1, 2)  # (B, 3, N)

        mod = F.relu(self.bn1(self.conv1(mod)))  # (B, 64, N)

        matrix64x64 = self.feature_transform(mod)  # (B, 64, 64)
        mod = torch.bmm(torch.transpose(mod, 1, 2), matrix64x64).transpose(1, 2)  # (B, 64, N)

        mod = F.relu(self.bn2(self.conv2(mod))) # (B, 128, N)
        mod = self.bn3(self.conv3(mod)) # (B, 1024, N)
        mod = nn.MaxPool1d(mod.size(-1))(mod) # (B, 1024, 1)
        output = nn.Flatten(1)(mod) # (B, 1024)
        return output, matrix3x3, matrix64x64


class TransformSymmetric(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def _get_transform_network(self, k):
        return TnetSymmetric(k=k)