import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mik_tools.mik_learning_tools.models.pointnet.t_net import Tnet
from mik_tools.mik_learning_tools.models.pointnet.transform_net import Transform, TransformSymmetric
from mik_tools.mik_learning_tools.models.aux import FCModule


class PointNetBase(nn.Module):
    def __init__(self, num_in_features, out_size, force_symmetric=True):
        super().__init__()
        self.num_in_features = num_in_features
        self.out_size = out_size
        self.force_symmetric = force_symmetric
        if self.force_symmetric:
            # here we force matrix3x3 and matrix64x64 to be symmetric by construction
            self.transform = TransformSymmetric(num_in_features=self.num_in_features)
        else:
            self.transform = Transform(num_in_features=self.num_in_features)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_size)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.criterion = torch.nn.MSELoss()

    def forward(self, input):
        # input shape: (B, num_points, 3)
        # output shape: (B,  out_size), (B, 3, 3), (B, 64, 64)
        mod, matrix3x3, matrix64x64 = self.transform(input)
        mod = F.relu(self.bn1(self.fc1(mod)))
        mod = F.relu(self.bn2(self.dropout(self.fc2(mod))))
        output = self.fc3(mod)
        return output, matrix3x3, matrix64x64

    def pointnetloss(self, outputs, labels, m3x3, m64x64, alpha=0.0001):
        bs = outputs.size(0) # batch size
        criterion = self.criterion
        if self.force_symmetric:
            return criterion(outputs, labels)
        else:
            # here, we add a loss term to enforce the matrices m3x3 and m64x64 to be symmetric
            new_output_size = outputs.size(0)
            id3x3 = torch.eye(3, requires_grad=True).repeat(new_output_size, 1, 1)
            id64x64 = torch.eye(64, requires_grad=True).repeat(new_output_size, 1, 1)
            if outputs.is_cuda:
                id3x3 = id3x3.cuda()
                id64x64 = id64x64.cuda()
            diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
            diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
            return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


class RegressionPointNet(PointNetBase):
    def __init__(self, num_in_features, out_size=10, force_symmetric=True):
        super().__init__(num_in_features=num_in_features, out_size=out_size, force_symmetric=force_symmetric)


class ClassificationPointNet(PointNetBase):
    def __init__(self, num_in_features, classes=10, force_symmetric=True):
        super().__init__(num_in_features=num_in_features, out_size=classes, force_symmetric=force_symmetric)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, input, softmax=True):
        output, matrix3x3, matrix64x64 = super().forward(input)
        if softmax:
            return self.logsoftmax(output), matrix3x3, matrix64x64
        else:
            return output, matrix3x3, matrix64x64


class PointNetEmbedding(nn.Module):
    def __init__(self, num_in_features, out_size, num_fcs=2):
        super().__init__()
        self.num_in_features = num_in_features
        self.out_size = out_size
        self.transform = TransformSymmetric(num_in_features=self.num_in_features) # This forces matrices matrix3x3 and matrix64x64 to be symmetric by construction.
        # Therefore, since marix3x3 and matrix64x64 are symmetric, no need for the loss function to enforce symmetry.
        fc_emb_sizes = [1024] + [1024*2]*(num_fcs-1) + [out_size]
        self.fc_emb = FCModule(sizes=fc_emb_sizes, activation='relu')
        self.criterion = torch.nn.MSELoss()

    def forward(self, input):
        # input shape: (..., num_points, 3)
        # output shape: (...,  out_size)
        input_shape = input.shape
        input = input.reshape(-1, *input_shape[-2:]) # (B, num_points, 3)
        mod, matrix3x3, matrix64x64 = self.transform(input) #(B, 1024), _, _
        output = self.fc_emb(mod) # (B, out_size)
        output = output.reshape(*input_shape[:-2] + (self.out_size,)) # (..., out_size)
        return output


class PointNetFeatureNetwork(nn.Module):
    def __init__(self, num_in_features=3, num_out_features=3, num_global_features=1024):
        super().__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_global_features = num_global_features
        self.input_transform = Tnet(k=self.num_in_features)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(self.num_in_features+self.num_global_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.out_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, input, global_features):
        # combine input and embedding by concatenating it ot each of the
        # input shape: (..., num_points, num_in_features)
        # gloabl_features (..., num_global_features)
        num_points = input.shape[-2]
        global_features_ext = global_features.unsqueeze(-2).repeat_interleave(num_points, -2)
        extended_input = torch.cat([input, global_features_ext], dim=-1)
        #Process the points
        matrix3x3 = self.input_transform(extended_input)
        # batch matrix multiplication
        mod = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        mod = F.relu(self.bn1(self.conv1(mod)))

        matrix64x64 = self.feature_transform(mod)
        mod = torch.bmm(torch.transpose(mod, 1, 2), matrix64x64).transpose(1, 2)

        mod = F.relu(self.bn2(self.conv2(mod)))
        output = self.conv3(mod)

        return output, matrix3x3, matrix64x64


class PointNetTrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.id3x3 = nn.Parameter(torch.eye(3), requires_grad=False)
        self.id64x64 = nn.Parameter(torch.eye(64), requires_grad=False)

    def forward(self, matrix3x3, matrix64x64):
        matrix3x3 = matrix3x3.flatten(end_dim=-3)
        matrix64x64 = matrix64x64.flatten(end_dim=-3)
        batch_size = matrix3x3.shape[0]
        id3x3 = self.id3x3.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        id64x64 = self.id64x64.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        diff3x3 = id3x3 - torch.bmm(matrix3x3, matrix3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(matrix64x64, matrix64x64.transpose(1, 2))
        pointnet_tr_loss = (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(batch_size)
        return pointnet_tr_loss






