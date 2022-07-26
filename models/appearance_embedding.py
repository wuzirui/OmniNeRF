import torch
from torch import nn


class AppearanceEmbedding(nn.Module):
    """
    Per-frame feature corrective latent code
    Reference:
    - NeRF in the Wild: Neural Radiance Fields for Unconstrained
        Photo Collection
    """
    def __init__(self, num_frames, num_channels) -> None:
        """
        num_frames: number of frames in the sequence
        num_channels: number of channels in the appearance embedding
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.embedding = nn.Parameter(
            torch.random.normal([num_frames, num_channels], dtype=torch.float32)
        )
    
    def forward(self, idx):
        idx = torch.where(idx < self.num_frames, idx, torch.zeros_like(idx))    # guard against out-of-bounds indices
        return torch.gather(self.embedding, idx)
    
    
