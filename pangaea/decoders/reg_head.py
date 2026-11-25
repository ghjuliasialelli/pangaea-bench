import torch
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


class RegHead(Decoder):
    """Base class for regression heads applied to the encoder's last feature map.

    The feature map is first resampled (bilinear, no parameters) to the target grid, so every
    encoder is decoded on the same grid whatever its patch size: only ``self.feature_dim``
    (the number of channels, 192..3072 depending on the encoder) differs, and only the head's
    first layer follows it. Heads are applied per pixel and implement
    ``head(feat) -> (B, 1, H, W)``. Single-temporal only, like RegUPerNet.
    """

    def __init__(self, encoder: Encoder, finetune: bool):
        super().__init__(encoder=encoder, num_classes=1, finetune=finetune)
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.feature_shape = self._probe_feature_shape()
        self.feature_dim = self.feature_shape[0]

    @torch.no_grad()
    def _probe_feature_shape(self) -> tuple[int, int, int]:
        """Run a dummy input through the encoder to get the (C, H, W) of its last output."""
        encoder = self.encoder
        was_training = encoder.training
        encoder.eval()
        device = next(encoder.parameters()).device
        dummy = {
            modality: torch.zeros(
                2, len(bands), 1, encoder.input_size, encoder.input_size, device=device
            )
            for modality, bands in encoder.input_bands.items()
        }
        try:
            feat = self._last_feature(encoder(dummy))
        except RuntimeError as e:
            raise ValueError(
                f"{type(self).__name__} is single-temporal, but {type(encoder).__name__} fails on a "
                f"(B, C, 1, H, W) input (multi-temporal encoder?): {e}"
            ) from e
        encoder.train(was_training)
        return tuple(feat.shape[1:])

    def _last_feature(self, feats: list[torch.Tensor]) -> torch.Tensor:
        feat = feats[-1]
        # multi-temporal encoders (e.g. Prithvi) keep a T=1 axis: (B, C, 1, H, W)
        if feat.dim() == 5 and feat.shape[2] == 1:
            feat = feat.squeeze(2)
        if feat.dim() != 4:
            raise ValueError(
                f"{type(self).__name__} expects a (B, C, H, W) feature map, got {tuple(feat.shape)}"
            )
        return feat

    def head(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        if self.finetune:
            feats = self.encoder(img)
        else:
            with torch.no_grad():
                feats = self.encoder(img)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # decode every encoder on the same grid: resample first, then run the head per pixel
        feat = self._last_feature(feats)
        if feat.shape[-2:] != tuple(output_shape):
            feat = F.interpolate(feat, size=output_shape, mode="bilinear", align_corners=False)

        output = self.head(feat)
        if output.shape[-2:] != tuple(output_shape):  # e.g. NicoNet with returns="pixel"
            output = F.interpolate(output, size=output_shape, mode="bilinear", align_corners=False)
        return output
