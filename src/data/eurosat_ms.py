"""Multispectral (13-band) EuroSAT data path via torchgeo.

Kept separate from the RGB `eurosat.py` loader so existing RGB runs are
unaffected. Band order is torchgeo's (verified from the downloaded data):

    idx  0    1    2    3    4    5    6    7    8    9    10   11   12
    band B01  B02  B03  B04  B05  B06  B07  B08  B09  B10  B11  B12  B8A
              Blue Grn  Red                NIR            SWIR1 SWIR2

Note B8A is LAST, not adjacent to B08.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 0-indexed band positions in torchgeo order.
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = 1, 2, 3, 7, 10, 11
CIRRUS = 9  # B10: atmospheric, no land signal; dropped by default.
N_BANDS = 13


def compute_indices(img: torch.Tensor) -> torch.Tensor:
    """Spectral indices from raw reflectance. img: [13, H, W] -> [3, H, W]."""
    eps = 1e-6
    red, nir, green, swir1 = img[RED], img[NIR], img[GREEN], img[SWIR1]
    ndvi = (nir - red) / (nir + red + eps)      # vegetation
    ndwi = (green - nir) / (green + nir + eps)  # water (River, SeaLake)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)  # built-up (Highway, Industrial, Residential)
    return torch.stack([ndvi, ndwi, ndbi], dim=0)


def _kept_bands(band_subset: str, drop_cirrus: bool):
    if band_subset == "rgb":
        return [RED, GREEN, BLUE]
    return [b for b in range(N_BANDS) if not (drop_cirrus and b == CIRRUS)]


def infer_in_channels(config: Dict) -> int:
    data = config["data"]
    if data.get("bands", "rgb") != "multispectral":
        return 3
    band_subset = data.get("band_subset", "all")
    n = len(_kept_bands(band_subset, data.get("drop_cirrus", True)))
    if band_subset != "rgb" and data.get("add_indices", False):
        n += 3
    return n


def _base_dataset(root: str, split: str):
    from torchgeo.datasets import EuroSAT as TGEuroSAT
    return TGEuroSAT(root=root, split=split, bands=TGEuroSAT.all_band_names, download=False)


def compute_or_load_stats(root: str, band_subset: str, drop_cirrus: bool, add_indices: bool) -> Dict[str, torch.Tensor]:
    """Per-channel mean/std over the train split. Cached to disk."""
    cache = Path(root) / f"ms_stats_{band_subset}_dropcirrus{int(drop_cirrus)}_indices{int(add_indices)}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: torch.tensor(z[k], dtype=torch.float32) for k in z.files}

    ds = _base_dataset(root, "train")
    keep = _kept_bands(band_subset, drop_cirrus)
    use_indices = add_indices and band_subset != "rgb"
    n_ch = len(keep) + (3 if use_indices else 0)
    # Welford-ish running sums over channels.
    total = torch.zeros(n_ch, dtype=torch.float64)
    total_sq = torch.zeros(n_ch, dtype=torch.float64)
    count = 0
    for i in range(len(ds)):
        img = ds[i]["image"].float()  # [13,H,W]
        chans = [img[keep]]
        if use_indices:
            chans.append(compute_indices(img))
        x = torch.cat(chans, dim=0)  # [n_ch,H,W]
        flat = x.reshape(n_ch, -1).double()
        total += flat.sum(dim=1)
        total_sq += (flat ** 2).sum(dim=1)
        count += flat.shape[1]
    mean = (total / count)
    var = (total_sq / count) - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-8))
    stats = {"mean": mean.float(), "std": std.float()}
    np.savez(cache, **{k: v.numpy() for k, v in stats.items()})
    return stats


class EuroSATMS(Dataset):
    def __init__(self, root: str, split: str, stats: Dict[str, torch.Tensor],
                 band_subset: str = "all", drop_cirrus: bool = True,
                 add_indices: bool = False, augment: bool = False):
        self.base = _base_dataset(root, split)
        self.keep = _kept_bands(band_subset, drop_cirrus)
        self.add_indices = add_indices and band_subset != "rgb"
        self.augment = augment
        self.mean = stats["mean"].reshape(-1, 1, 1)
        self.std = stats["std"].reshape(-1, 1, 1)

    def __len__(self):
        return len(self.base)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Geometry-only augmentation (flips/90-degree rotations) — safe for all channels.
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
        return x

    def __getitem__(self, index):
        sample = self.base[index]
        img = sample["image"].float()  # [13,H,W] raw reflectance
        label = int(sample["label"])
        chans = [img[self.keep]]
        if self.add_indices:
            chans.append(compute_indices(img))  # computed from full raw bands
        x = torch.cat(chans, dim=0)
        x = (x - self.mean) / self.std
        if self.augment:
            x = self._augment(x)
        return x, label


def build_ms_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    data = config["data"]
    root = data.get("root_ms", "data/raw/EuroSATMS")
    band_subset = data.get("band_subset", "all")
    drop_cirrus = data.get("drop_cirrus", True)
    add_indices = data.get("add_indices", False)
    batch_size = data["batch_size"]
    num_workers = data.get("num_workers", 4)

    stats = compute_or_load_stats(root, band_subset, drop_cirrus, add_indices)

    loaders = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        ds = EuroSATMS(root, split, stats, band_subset, drop_cirrus, add_indices, augment=aug)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True,
        )
    return loaders
