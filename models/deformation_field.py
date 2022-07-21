import torch
from torch import nn


class DeformationField(nn.Module):
    def __init__(self, D=6, W=128, in_channels=2, out_channels=2, skips=[3]):
        """
        D: number of layers for deformation fields
        W: number of hidden units in each layer
        in_channels: number of input channels for the MLP (2 by default)
        out_channels: number of output channels for the MLP (2 by default)
        skips: add skip connection in the Dth layer (zero-indexed)
        """
        super(DeformationField, self).__init__()
        self.D, self.W, self.in_channels, self.out_channels, self.skips = \
            D, W, in_channels, out_channels, skips
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"deformation_field_{i+1}", layer)


    def forward(self, x):
        """
        Predicts the pose refinement of the input pose.

        Inputs:
            x: (B, in_channels), where B is the batch size
        Outputs:
            out: (B, out_channels)
        """
        out = x
        for i in range(self.D):
            layer = getattr(self, f"deformation_field_{i+1}")
            if i in self.skips:
                out = torch.cat([out, x], -1)
            out = layer(out)
        return out