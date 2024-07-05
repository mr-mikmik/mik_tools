import numpy as np
import torch
import torch.nn as nn

from mik_tools import transform_points_3d, tr, pose_to_matrix, matrix_to_pose


class ModelPoseLoss(torch.nn.Module):
    def __init__(self, model_points: torch.Tensor, criterion=None):
        super().__init__()
        self.criterion = criterion
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
            losses (torch.Tensor): of shape (...,)
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


if __name__ == '__main__':
    # Debug
    debug_pose_loss()