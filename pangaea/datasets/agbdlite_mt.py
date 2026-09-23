"""Multi-temporal AGBD-Lite.

Three Sentinel-2 L2A observations per GEDI footprint instead of one. This is a separate class from
AGBDLite on purpose: the single-temporal path is used by every published run in the benchmark, and
nothing here changes it. The only seam in the parent is that the file prefix is read from
`self.fname_prefix` when set, defaulting to the original 'AGBD-Lite'.

The files are produced by AGBD-GFMs/data/agbd_lite/gen_multitemporal.py and differ from AGBD-Lite in
exactly three datasets:

    S2_bands                        (N, 3, 25, 25, 12)  instead of (N, 25, 25, 12)
    Sentinel_metadata/S2_date       (N, 3)              instead of (N,)
    Sentinel_metadata/S2_boa_offset (N, 3)              instead of (N,)

Everything else -- footprints, split membership, targets, ALOS, DEM, LC -- is identical, and the
Sentinel-2 patch used by AGBD-Lite is present bit-for-bit in one of the three slots, so results stay
comparable with the single-temporal runs.
"""

import numpy as np
import torch

from pangaea.datasets.agbd import ALOS_BANDS, alos_dn_to_db
from pangaea.datasets.agbdlite import AGBDLite


class AGBDLiteMT(AGBDLite):
    def __init__(self, fname_prefix: str = 'AGBD-Lite-MT3', **kwargs):
        # Set before super().__init__ so the parent picks it up when it resolves the filename.
        self.fname_prefix = fname_prefix
        super(AGBDLiteMT, self).__init__(**kwargs)

        with_t = self.f_handle['S2_bands'].ndim == 5
        assert with_t, (
            f"{self.fname} has S2_bands of shape {self.f_handle['S2_bands'].shape}; "
            f"AGBDLiteMT expects (N, T, H, W, C). Use dataset=agbdlite for single-temporal files."
        )
        self.n_timesteps = self.f_handle['S2_bands'].shape[1]
        assert self.n_timesteps == self.multi_temporal, (
            f"{self.fname} has {self.n_timesteps} timesteps but the config says "
            f"multi_temporal={self.multi_temporal}; the encoder would be built for the wrong T."
        )

    def __getitem__(self, n):
        """Returns the i-th item of the dataset.

        Returns:
            dict with 'image' {'optical': (C, T, H, W), 'sar': (C, T, H, W)}, 'target' (H, W) and
            'metadata'. The optical timesteps are ordered by acquisition date.
        """
        idx_start = n * self.lite_chunk_size
        idx_end = min(idx_start + self.lite_chunk_size, self.gedi_length)
        f = self.f_handle

        # Sentinel-2 bands ------------------------------------------------------------------------
        if not hasattr(self, 's2_order') : self.s2_order = list(f['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.s2_bands]

        # (B, T, H, W, C)
        s2_bands = f['S2_bands'][idx_start : idx_end, :, :, :, self.s2_indices].astype(np.float32)

        # The BOA offset is per timestep here, because the three dates can come from different
        # processing baselines. Applying a single sample-level offset would shift two of them.
        if 'S2_boa_offset' in f['Sentinel_metadata'].keys() :
            s2_boa_offset = f['Sentinel_metadata']['S2_boa_offset'][idx_start : idx_end].astype(np.float32)
        else: s2_boa_offset = np.zeros(s2_bands.shape[:2], dtype = np.float32)
        if s2_boa_offset.ndim == 1 : s2_boa_offset = np.repeat(s2_boa_offset[:, None], s2_bands.shape[1], axis = 1)
        s2_boa_offset = s2_boa_offset[:, :, np.newaxis, np.newaxis, np.newaxis]

        sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
        sr_bands[s2_bands == 0] = 0
        sr_bands[sr_bands < 0] = 0

        # SAR bands (from ALOS-PALSAR-2) ----------------------------------------------------------
        if not hasattr(self, 'alos_order') : self.alos_order = list(f['ALOS_bands'].attrs['order'])
        if not hasattr(self, 'alos_indices') : self.alos_indices = [self.alos_order.index(band) for band in ALOS_BANDS]

        alos_bands = f['ALOS_bands'][idx_start : idx_end, :, :, self.alos_indices]
        alos_bands = alos_dn_to_db(alos_bands, self.data_min['sar'], self.data_max['sar'])

        # Target data -----------------------------------------------------------------------------
        target_value = torch.from_numpy(np.array(f['GEDI'][self.target][idx_start : idx_end], dtype = np.float32)).to(torch.float)
        lc = torch.from_numpy(np.array(f['LC'][idx_start : idx_end, :, :, 0])).long()
        target = torch.full_like(lc, fill_value = self.ignore_index, dtype = torch.float if self.target in ['agbd', 'rh98'] else torch.long)
        target[:, self.patch_size // 2, self.patch_size // 2] = target_value

        # Metadata ---------------------------------------------------------------------------------
        region = torch.from_numpy(np.array(f['GEDI']['region_cla'][idx_start : idx_end])).long()
        biome = lc[:, self.patch_size // 2, self.patch_size // 2]
        s2_date = torch.from_numpy(np.array(f['Sentinel_metadata']['S2_date'][idx_start : idx_end])).long()

        # Convert to tensors and return --------------------------------------------------------------
        sr_bands = torch.from_numpy(sr_bands).float()
        sr_bands = sr_bands.permute(0, 4, 1, 2, 3)          # (B, T, H, W, C) -> (B, C, T, H, W)
        alos_bands = torch.from_numpy(alos_bands).float()
        alos_bands = alos_bands.permute(0, 3, 1, 2).unsqueeze(2)   # (B, C, 1, H, W)
        # ALOS-PALSAR-2 is a yearly mosaic: there is ONE observation per footprint, not one per
        # Sentinel-2 date. It is broadcast across T so that a MULTIMODAL encoder sees matching
        # temporal axes -- these are not three independent SAR acquisitions.
        #
        # None of the three encoders this dataset was added for (prithvi, prithvi2_100m,
        # satlasnet_mi) declare a `sar` modality, so BandFilter drops this tensor before it reaches
        # them and the broadcast is a no-op for those runs. It is kept, as a stride-0 view rather
        # than a copy, so that croma_joint / terramind_tiny / dofa_joint also work against this
        # dataset instead of failing on a T mismatch.
        alos_bands = alos_bands.expand(-1, -1, self.n_timesteps, -1, -1)

        # TEMPORARY, until we can return chunks (mirrors AGBDLite)
        sr_bands = sr_bands.squeeze(0)
        alos_bands = alos_bands.squeeze(0)
        target = target.squeeze(0)
        region = region.squeeze(0)
        biome = biome.squeeze(0)
        s2_date = s2_date.squeeze(0)

        return {
            'image': {
                'optical': sr_bands,
                'sar': alos_bands
                },
            'target': target,
            'metadata': {
                'region': region,
                'biome': biome,
                's2_date': s2_date,
            }
        }
