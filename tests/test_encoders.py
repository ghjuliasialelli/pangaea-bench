import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from pangaea.encoders.base import Encoder


# These tests compose a single encoder config in isolation, so there is no `dataset` node
# to resolve against. Every encoder whose config interpolates ${dataset.*} therefore
# raises InterpolationKeyError here and is excluded -- that, and only that, is why the
# list below is not the full contents of configs/encoder/:
#
#   dofa, dofa_optical, dofa_joint   ${dataset.bands}
#   prithvi, prithvi2_100m           ${dataset.multi_temporal}
#   unet_encoder, unet_encoder_mi    ${dataset.img_size} (+ bands, multi_temporal)
#   resnet50_*, vit_mi, vit_scratch  ${dataset.img_size} / ${dataset.bands}
#
# To cover those too, compose configs/ with `dataset=<name>` rather than configs/encoder/
# alone. Anything without an interpolation belongs in the list; check with
#   grep -l '${dataset' configs/encoder/*.yaml
@pytest.mark.parametrize(
    "config_name",
    [
        "croma_joint",
        "croma_optical",
        "croma_sar",
        "gfmswin",
        "remoteclip",
        "satlasnet_si",
        "satlasnet_mi",
        "scalemae",
        "spectralgpt",
        "ssl4eo_data2vec",
        "ssl4eo_dino",
        "ssl4eo_mae_optical",
        "ssl4eo_mae_sar",
        "ssl4eo_moco",
        "terramind_optical_tiny",
        "terramind_tiny",
        "thor",
        "vit",
    ],
)
def test_encoder_init(config_name: str) -> None:
    with initialize(version_base=None, config_path="../configs/encoder/"):
        # config is relative to a module
        encoder_config = compose(config_name=config_name)
        encoder = instantiate(encoder_config)
        assert isinstance(encoder, Encoder)


@pytest.mark.parametrize(
    "config_name",
    [
        "croma_joint",
        "croma_optical",
        "croma_sar",
        "gfmswin",
        "remoteclip",
        "satlasnet_si",
        "satlasnet_mi",
        "scalemae",
        "spectralgpt",
        "ssl4eo_data2vec",
        "ssl4eo_dino",
        "ssl4eo_mae_optical",
        "ssl4eo_mae_sar",
        "ssl4eo_moco",
        "terramind_optical_tiny",
        "terramind_tiny",
        "thor",
        "vit",
    ],
)
def test_encoder_input_shape(config_name: str) -> None:
    with initialize(version_base=None, config_path="../configs/encoder/"):
        # config is relative to a module
        encoder_config = compose(config_name=config_name)
        encoder = instantiate(encoder_config)

        # fake data
        B, T = 2, 1
        data = {}
        for modality, bands in encoder.input_bands.items():
            n_bands = len(bands)
            H = W = encoder.input_size
            if encoder.multi_temporal:
                data[modality] = torch.randn(B, n_bands, T, H, W)
            else:
                data[modality] = torch.randn(B, n_bands, H, W)

