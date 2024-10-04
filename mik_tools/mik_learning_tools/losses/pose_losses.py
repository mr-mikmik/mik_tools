import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mik_tools import transform_points_3d, tr, pose_to_matrix, matrix_to_pose


class ModelPoseLoss(torch.nn.Module):
    def __init__(self, model_points: torch.Tensor, criterion=None):
        super().__init__()
        if criterion == 'mean_dist':
            self.criterion = mean_dist_criterion
        elif criterion == 'mse':
            self.criterion = mse_criterion
        elif criterion is None:
            self.criterion = mse_criterion
        elif criterion is callable:
            self.criterion = criterion
        else:
            raise ValueError(f'Criterion {criterion} not recognized. Please select one of the following: [None, "mean_dist", "mse"] or pass a callable function')
        assert len(model_points.shape) == 2 and model_points.shape[-1] == 3, f'Model points must have shape (N,3) (given {model_points.shape} not valid)'
        self.model_points = nn.Parameter(model_points, requires_grad=False) # store the points as parameter to have them automatically set on the model device
        self.num_points = self.model_points.shape[0]

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between two sets of poses according to the object model.
        The loss is computed as the criterion based on the point-to-point correspondences between the transformed self.model points
        by thefault, it is the MSE of the point-to-point correspondences between the transformed self.model points
        Args:
            X_1 (torch.Tensor): of shape (..., 4, 4) representing the SE(3) pose 1
            X_2 (torch.Tensor): of shape (..., 4, 4) representing the SE(3) pose 2
        Returns:
            losses (torch.Tensor): of shape (..., )
        """
        # import pdb; pdb.set_trace()
        points_1 = transform_points_3d(self.model_points, X_1, only_transform_are_batched=True) # (..., N, 3)
        points_2 = transform_points_3d(self.model_points, X_2, only_transform_are_batched=True) # (..., N, 3)
        if self.criterion is None:
            # compute the MSE between the set of points -- not reduce it
            points_loss = torch.pow(points_2 - points_1, 2).mean(dim=-1) # (..., N) by default (no-reduction on batch dims)
            loss = points_loss.mean(dim=-1) # (...,)
        else:
            loss = self.criterion(points_1, points_2)
        return loss

    def batched_pose_loss(self, X_1: torch.Tensor, X_2: torch.Tensor, batch_size=None) -> torch.Tensor:
        """
        Compute the loss between two differently batched sets of poses according to the object model.
        Note: This is highly computational expensive since we need to compute the loss for each element in 1 to each element in 2
        Args:
            X_1 (torch.Tensor): of shape (B, 4, 4) representing the SE(3) pose 1
            X_2 (torch.Tensor): of shape (K, 4, 4) representing the SE(3) pose 2
        Returns:
            losses (torch.Tensor): of shape (B, K) #loss for each pose pair in the provided tensors
        """
        B = X_1.shape[0]
        K = X_2.shape[0]
        losses = []
        if batch_size is None:
            batch_size = B
        num_batches = B // batch_size
        last_batch_size = B - num_batches*batch_size
        X_2_augmented = X_2.unsqueeze(dim=0).repeat_interleave(repeats=batch_size, dim=0)# (batch_size, K, 4, 4)
        for i in tqdm(range(num_batches), desc='batched_pose_loss'):
            X_1_augmented = X_1[i*batch_size:(i+1)*batch_size].unsqueeze(dim=1).repeat_interleave(repeats=K, dim=1)# (batch_size, K, 4, 4)
            loss = self.forward(X_1_augmented, X_2_augmented) # (batch_size, K)
            losses.append(loss)
        # last
        if last_batch_size > 0:
            X_2_augmented = X_2.unsqueeze(dim=0).repeat_interleave(repeats=last_batch_size, dim=0)  # (last_batch_size, K, 4, 4)
            X_1_augmented = X_1[-last_batch_size:].unsqueeze(dim=1).repeat_interleave(repeats=K,dim=1)  # (last_batch_size, K, 4, 4)
            loss = self.forward(X_1_augmented, X_2_augmented)  # (last_batch_size, K)
            losses.append(loss)
        losses = torch.cat(losses, dim=0) # (B, K)
        return losses


def mean_dist_criterion(points_1, points_2):
    # points_1: (..., N, 3)
    # points_2: (..., N, 3)
    # out: (...)
    delta_dists = torch.sqrt(torch.pow(points_2 - points_1, 2).sum(dim=-1))  # (..., N)
    loss = delta_dists.mean(dim=-1)  # (...)
    return loss


def mse_criterion(points_1, points_2):
    # points_1: (..., N, 3)
    # points_2: (..., N, 3)
    # out: (...)
    delta_dists = torch.pow(points_2 - points_1, 2).sum(dim=-1)  # (..., N)
    loss = delta_dists.mean(dim=-1)  # (...)
    return loss



def debug_pose_loss():
    model_points = torch.tensor(np.random.uniform(-10,10, (50,3)), dtype=torch.float32)
    pose_loss = ModelPoseLoss(model_points)
    pose_1 = torch.zeros((10, 7))
    pose_1[...,-1] = 1
    pose_2 = pose_1.clone()
    X_1 = pose_to_matrix(pose_1)
    X_2 = pose_to_matrix(pose_2)
    loss_value = pose_loss(X_1, X_2)
    print('Loss value:', loss_value)


def debug_batched_pose_loss():
    from mik_tools import tr
    model_points = torch.tensor(np.random.uniform(-10, 10, (50, 3)), dtype=torch.float32)
    pose_loss = ModelPoseLoss(model_points)
    X_1 = tr.random_transform_matrix(shape=50, as_tensor=True)
    X_2 = tr.random_transform_matrix(shape=20, as_tensor=True)
    loss_value = pose_loss.batched_pose_loss(X_1, X_2)
    print(loss_value.shape)


if __name__ == '__main__':
    # Debug
    debug_pose_loss()
    debug_batched_pose_loss()