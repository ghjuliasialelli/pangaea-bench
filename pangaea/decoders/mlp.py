import torch
import torch.nn as nn

from pangaea.decoders.reg_head import RegHead
from pangaea.encoders.base import Encoder


class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(MLP, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            self.relu,            
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.relu,
            
            nn.Linear(hidden_dim // 2, output_dim),
            self.relu
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, output_dim, H, W): the network is applied per pixel
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, -1)
        return self.network(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)


class RegMLP(RegHead):
    """Per-pixel MLP on the encoder's features, resampled to the target grid."""

    def __init__(self, encoder: Encoder, finetune: bool, hidden_dim: int = 128):
        super().__init__(encoder=encoder, finetune=finetune)
        self.model_name = "Reg_MLP"
        self.mlp = MLP(input_dim=self.feature_dim, hidden_dim=hidden_dim, output_dim=1)

    def head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)
