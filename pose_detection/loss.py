import torch
import torch.nn as nn


class PoseEstimationLoss(nn.Module):

    def __init__(self):
        super(PoseEstimationLoss, self).__init__()
        self.gaussian_loss = nn.GaussianNLLLoss(eps=1e-1)# was 0.3
        self.angle_constant = 1
        self.pos_constant = 1

    def forward(self, predictions, target):
        x_loss = self.gaussian_loss(
            predictions[..., 0:1]*self.pos_constant,
            target[..., 0:1]*self.pos_constant,
            torch.square(predictions[..., 4:5]*self.pos_constant)
        )
        y_loss = self.gaussian_loss(
            predictions[..., 1:2]*self.pos_constant,
            target[..., 1:2]*self.pos_constant,
            torch.square(predictions[..., 5:6]*self.pos_constant)
        )
        cos_yaw_loss = self.gaussian_loss(
            predictions[..., 2:3]*self.angle_constant,
            target[..., 2:3] * self.angle_constant,
            torch.square(predictions[..., 6:7] * self.angle_constant)
        )
        sin_yaw_loss = self.gaussian_loss(
            predictions[..., 3:4]*self.angle_constant,
            target[..., 3:4] * self.angle_constant,
            torch.square(predictions[..., 7:8] * self.angle_constant)
        )

        loss = (
            x_loss + y_loss + cos_yaw_loss + sin_yaw_loss
        )
        return torch.mean(loss)

