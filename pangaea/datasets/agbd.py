import torch
from pangaea.datasets.base import RawGeoFMDataset

import numpy as np
import h5py
from os.path import join
import pickle
import os

ALOS_BANDS = ['HH', 'HV']


def alos_dn_to_db(dn, db_min, db_max):
    """
    Convert ALOS-PALSAR-2 digital numbers to gamma naught (dB), clipped to [db_min, db_max].

    DN 0 and DN 1 (the latter maps to exactly -83 dB) are nodata; they are set to `db_min`, i.e. to 0
    after min-max normalisation, the same value BandPadding uses for missing bands.

    Args:
    - dn (np.ndarray): DN values, of shape (..., n_bands)
    - db_min (list): per-band lower clipping bound, in dB
    - db_max (list): per-band upper clipping bound, in dB
    Returns:
    - db (np.ndarray): float32 gamma naught values, of the same shape as `dn`
    """
    db_min = np.asarray(list(db_min), dtype = np.float32)
    db_max = np.asarray(list(db_max), dtype = np.float32)
    nodata = dn <= 1
    db = 20 * np.log10(np.maximum(dn, 1).astype(np.float32)) - 83.0
    db = np.clip(db, db_min, db_max)
    return np.where(nodata, db_min, db).astype(np.float32)


continent_to_region = {'North America': ['California', 'Cuba'], 'South America': ['Paraguay', 'FrenchGuiana'],
    'Africa': ['UnitedRepublicofTanzania', 'Ghana'], 'Europe': ['Austria', 'Greece'],
    'South Asia': ['Nepal', 'ShaanxiProvince'], 'Australasia': ['NewZealand']}

def region_tiles(path_h5, region) :
    """The S2 tiles of a world region (a key of `continent_to_region`, e.g. 'South Asia'), from tiles_per_region.pkl --
    the same definition as AGBD-GFMs/model/dataset.py (whose keys are the same without spaces, e.g. 'SouthAsia')."""
    if region not in continent_to_region :
        raise ValueError(f"Unknown region {region!r}; expected one of {list(continent_to_region)}")
    with open(join(path_h5, 'tiles_per_region.pkl'), 'rb') as f: tiles_per_region = pickle.load(f)
    return set(str(t) for subregion in continent_to_region[region] for t in tiles_per_region[subregion])


def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5, hold_out_region = None, keep_region = False, drop_overlaps = False, test_region = None) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.

    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - chunk_size (int): the size of the chunks
    - path_mapping (str): the path to the file mapping each mode to its tiles
    - path_h5 (str): the path to the h5 files
    - hold_out_region (str): the region to hold out
    - keep_region (bool): whether to keep the specified region
    - drop_overlaps (bool): whether to drop overlapping patches
    - test_region (str): in test mode, keep ONLY the tiles of this region (regional test)

    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)[mode]

    # Region selection. train / val: drop the tiles of `hold_out_region` (keep_region = False, cross-region) or keep
    # only them (keep_region = True, within-region). test: keep only the tiles of `test_region`.
    if hold_out_region and mode in ['train', 'val'] : sel_tiles, keep = region_tiles(path_h5, hold_out_region), keep_region
    elif test_region and mode == 'test' : sel_tiles, keep = region_tiles(path_h5, test_region), True
    else : sel_tiles, keep = None, None

    # If need to drop the test patches that overlap with the AEF train set
    if drop_overlaps:
        with open(join(path_h5, 'AEF_overlaps.pkl'), 'rb') as f:
            overlap = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames :
        idx[fname] = {}
        
        with h5py.File(join(path_h5, fname), 'r') as f:
            
            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping)
            
            # Iterate over the tiles
            for tile in tiles :

                if sel_tiles is not None and (str(tile) in sel_tiles) != keep : continue

                # Get the number of patches in the tile
                if drop_overlaps :
                    if fname in overlap and tile in overlap[fname] : indices_to_skip = overlap[fname][tile]
                    else: indices_to_skip = []
                    n_total = len(f[tile]['GEDI']['agbd'])
                    n_patches = n_total - len(indices_to_skip)
                    idx[fname][tile] = {'n_patches' : n_patches, 'n_total' : n_total, 'indices_to_skip' : indices_to_skip}
                else:
                    n_patches = len(f[tile]['GEDI']['agbd'])
                    idx[fname][tile] = n_patches // chunk_size
    
    if drop_overlaps : total_length = sum(sum(d['n_patches'] for d in idx[fname].values()) for fname in idx.keys())
    else: total_length = sum(sum(v for v in d.values()) for d in idx.values())
    if sel_tiles is not None :
        assert total_length > 0, f'AGBD {mode}: no samples left after the region selection'
        print(f'AGBD {mode}: ' + (f'test region {test_region}' if mode == 'test' else f"{'keeping only' if keep else 'holding out'} {hold_out_region}") + f' -> {total_length:,} samples')

    return idx, total_length


def init_ranges_for_chunk(index, total_length, drop_overlaps = False):
    """
    This function creates a list of tuples (start_idx, end_idx, fname, tname) for each tile in the index, where
    start_idx and end_idx are the indices of the first and last chunk of the tile in the dataset. This will allow us to
    quickly find the file, tile, and row index corresponding to a given chunk index.

    Args:
    - index (dict): the index of the dataset, mapping file names to tile names and tile names to number of chunks
    - total_length (int): the total number of chunks in the dataset
    - oversampling (bool): whether to use oversampling or not
    - drop_overlaps (bool): whether to drop overlapping patches or not

    Returns:
    - ranges (list): list of tuples (start_idx, end_idx, fname, tname) for each tile in the index
    """

    ranges = []
    start_idx = 0
    for fname, file_data in index.items() :
        for tname, tile_data in file_data.items() :
            num_patches = tile_data['n_patches'] if drop_overlaps else tile_data
            end_idx = start_idx + num_patches
            assert end_idx <= total_length, f"Index out of bounds: {end_idx} > {total_length}"
            ranges.append((start_idx, end_idx, fname, tname))
            start_idx = end_idx

    return ranges


def find_index_for_chunk(index, ranges, n, total_length, drop_overlaps = False) :
    """
    For a given `index`, `ranges`, and `n`-th chunk, find the file, tile, and row index corresponding to this chunk.
    
    Args:
    - index (dict): dictionary mapping the files to the tiles and the tiles to the chunks
    - ranges (list): list of tuples (start_idx, end_idx, fname, tname) for each tile in the index
    - n (int): the n-th chunk
    - total_length (int): the total number of chunks in the dataset
    - chunk_size (int): the size of the chunks
    - oversampling (bool): whether to use oversampling or not
    - lite (bool): whether to use the lite version of the dataset
    - drop_overlaps (bool): whether to drop overlapping patches or not

    Returns:
    - file_name (str): the name of the file
    - tile_name (str): the name of the tile
    - chunk_within_tile (int): the chunk index within the tile
    """

    # Check that the chunk index is within bounds
    assert n < total_length, "The chunk index is out of bounds"

    for start, end, fname, tname in ranges :
        if start <= n < end :
            chunk_within_tile = n - start
            
            if drop_overlaps :
                tile_data = index[fname][tname]
                indices_to_skip = tile_data['indices_to_skip']
                n_total = tile_data['n_total']
                indices_to_keep = np.setdiff1d(np.arange(n_total), indices_to_skip)
                chunk_within_tile = indices_to_keep[chunk_within_tile]

            return fname, tname, chunk_within_tile


def get_file_and_idx(eval_lite, mode, index, ranges, n, length, drop_overlaps, handles, rows = None) :

    if eval_lite and mode == 'val' :
        row = n if rows is None else int(rows[n]) # rows: the AGBD-Lite-val rows kept by the region selection
        idx_start, idx_end = row, row + 1
        return handles, idx_start, idx_end

    else:
        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx = find_index_for_chunk(index, ranges, n, length, drop_overlaps)
        f = handles[file_name][tile_name]
        idx_start, idx_end = idx, idx + 1
        return f, idx_start, idx_end



class AGBD(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        root_path_cluster: str,
        target: str,
        hold_out_region: str | None = None,
        keep_region: bool = False,
        drop_overlaps: bool = False,
        eval_lite: bool = False,
        test_region: str | None = None,
        continents_file: str = 'AGBD-Lite-continents.npz',
        years: list[int] | None = None,
        test_years: list[int] | None = None
    ):
        super(AGBD, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download
        )

        assert split in ['train', 'val', 'test'], "split must be one of 'train', 'val', or 'test'"
        self.mode = split
        self.eval_lite = eval_lite
        self.target = target
        self.patch_size = img_size
        self.s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        if os.environ.get('SLURM_SUBMIT_DIR') is not None:
            print('Running on cluster, using cluster root path.')
            self.root_path = root_path_cluster
        else: self.root_path = root_path
        self.h5_path, self.mapping = self.root_path, self.root_path

        # Evaluate on the AGBD-Lite validation set
        self.rows = None
        if self.eval_lite and split == 'val' :
            self.handles = h5py.File(join(self.root_path, 'AGBD-Lite-val.h5'), 'r')
            self.index, self.ranges, self.drop_overlaps = None, None, None
            self.length = len(self.handles['GEDI']['agbd'])
            # The validation set picks the best checkpoint, so it gets the same region selection as train: without
            # it, a cross-region run would select its checkpoint on data that includes the held-out region.
            if hold_out_region :
                from pangaea.datasets.agbdlite import agbd_fingerprint, lite_rows_for_region # (agbdlite imports agbd)
                self.rows = lite_rows_for_region(join(self.root_path, continents_file), 'val', hold_out_region.replace(' ', ''),
                                                 keep_region = keep_region, n_rows = self.length,
                                                 fingerprint = agbd_fingerprint(self.handles['GEDI']['agbd'][:]))
                self.length = len(self.rows)

        # Regular training and evaluation on the full dataset
        else:
            # Temporal generalization: `years` for train (and val, when not eval_lite), `test_years` for test.
            years = list(test_years if split == 'test' and test_years is not None else years if years is not None else [2019, 2020])
            assert set(years) <= {2019, 2020} and len(years) > 0, f'years must be a non-empty subset of [2019, 2020], got {years}'
            self.fnames = [f'data_subset-{year}-v4_{i}-20.h5' for i in range(20) for year in [y for y in [2019, 2020] if y in years]]

            self.hold_out_region = hold_out_region
            self.keep_region = keep_region
            self.drop_overlaps = drop_overlaps and split == 'test'

            self.index, self.length = initialize_index(self.fnames, self.mode, 1, self.mapping, self.h5_path, self.hold_out_region, self.keep_region, self.drop_overlaps, test_region)
            self.ranges = init_ranges_for_chunk(self.index, self.length, drop_overlaps = self.drop_overlaps)

            self.handles = {fname: h5py.File(join(self.h5_path, fname), 'r') for fname in self.index.keys()}


    def __len__(self):
        # Return the total number of samples
        return int(self.length)

    def __getitem__(self, n):
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                    "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                    },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
                "metadata": dict}.
        """

        # Find the file, tile, and row index corresponding to this chunk
        f, idx_start, idx_end = get_file_and_idx(self.eval_lite, self.mode, self.index, self.ranges, n, self.length, self.drop_overlaps, self.handles, self.rows)

        # Sentinel-2 bands ------------------------------------------------------------------------
        
        # Set the order and indices for the Sentinel-2 bands
        if not hasattr(self, 's2_order') : self.s2_order = list(f['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.s2_bands]

        # Get the bands
        s2_bands = f['S2_bands'][idx_start : idx_end, :, :, self.s2_indices].astype(np.float32)

        # Get the BOA offset, if it exists
        if 'S2_boa_offset' in f['Sentinel_metadata'].keys() : s2_boa_offset = f['Sentinel_metadata']['S2_boa_offset'][idx_start : idx_end].astype(np.float32)
        else: s2_boa_offset = np.full((s2_bands.shape[0],), 0, dtype = np.float32)
        s2_boa_offset = s2_boa_offset[:, np.newaxis, np.newaxis, np.newaxis]

        # Get the surface reflectance values
        sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
        sr_bands[s2_bands == 0] = 0
        sr_bands[sr_bands < 0] = 0

        # SAR bands (from ALOS-PALSAR-2) ----------------------------------------------------------

        # Get the bands as clipped gamma naught values
        alos_order = list(f['ALOS_bands'].attrs['order'])
        alos_indices = [alos_order.index(band) for band in ALOS_BANDS]
        alos_bands = f['ALOS_bands'][idx_start : idx_end, :, :, alos_indices]
        alos_bands = alos_dn_to_db(alos_bands, self.data_min['sar'], self.data_max['sar'])

        # Target data -----------------------------------------------------------------------------
        target_value = torch.from_numpy(np.array(f['GEDI'][self.target][idx_start : idx_end], dtype = np.float32)).to(torch.float)
        lc = torch.from_numpy(np.array(f['LC'][idx_start : idx_end, :, :, 0])).long()
        target = torch.full_like(lc, fill_value = self.ignore_index, dtype = torch.float if self.target in ['agbd', 'rh98'] else torch.long)
        target[:, self.patch_size // 2, self.patch_size // 2] = target_value
        
        # Metadata (if needed) --------------------------------------------------------------------
        region = torch.from_numpy(np.array(f['GEDI']['region_cla'][idx_start : idx_end])).long()
        biome = lc[:, self.patch_size // 2, self.patch_size // 2]

        # Convert to tensors and return -----------------------------------------------------------
        sr_bands = torch.from_numpy(sr_bands).float()
        sr_bands = sr_bands.permute(0, 3, 1, 2).unsqueeze(2) # Change to (B, C, 1, H, W)
        alos_bands = torch.from_numpy(alos_bands).float()
        alos_bands = alos_bands.permute(0, 3, 1, 2).unsqueeze(2) # Change to (B, C, 1, H, W)

        # TEMPORARY, until we can return chunks
        sr_bands = sr_bands.squeeze(0)
        alos_bands = alos_bands.squeeze(0)
        target = target.squeeze(0)
        region = region.squeeze(0)
        biome = biome.squeeze(0)

        return {
            'image': {
                'optical': sr_bands,
                'sar': alos_bands
                },
            'target': target,
            'metadata': {
                'region': region,
                'biome': biome
            }
        }

    @staticmethod
    def download(self, silent=False):
        pass