import torch
import torch.nn as nn

from pangaea.decoders.reg_head import RegHead
from pangaea.encoders.base import Encoder


class LP(nn.Module):
    def __init__(self, input_dim=64, output_dim=1):
        super(LP, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        
        self.lp = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # (B, C, H, W) -> (B, output_dim, H, W): the linear layer is applied per pixel
        x = x.permute(0, 2, 3, 1)
        return self.lp(x).permute(0, 3, 1, 2)


class RegLP(RegHead):
    """Per-pixel linear probe on the encoder's features, resampled to the target grid."""

    def __init__(self, encoder: Encoder, finetune: bool):
        super().__init__(encoder=encoder, finetune=finetune)
        self.model_name = "Reg_LP"
        self.lp = LP(input_dim=self.feature_dim, output_dim=1)

    def head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.lp(feat)
