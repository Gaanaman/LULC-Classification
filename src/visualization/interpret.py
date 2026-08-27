"""Interpretability helpers: Grad-CAM, class spectra, and band composites.

Grad-CAM is implemented directly with forward/backward hooks so the package
does not depend on an external CAM library. The target layer is the ReLU of a
chosen conv block, which is where the spatial feature map still has resolution.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.eurosat_ms import RED, GREEN, BLUE, NIR, SWIR1, SWIR2, _base_dataset
from src.models.spectral import build_scratch_from_config, VirtualSensorNet
from src.utils.io import read_yaml

# Three-band composites over the raw 13-band stack. Each is a standard
# Sentinel-2 rendering; the false-colour ones show signal RGB cannot carry.
COMPOSITES: Dict[str, Tuple[int, int, int]] = {
    "True colour (B04/B03/B02)": (RED, GREEN, BLUE),
    "False colour infrared (B08/B04/B03)": (NIR, RED, GREEN),
    "Agriculture (B11/B08/B02)": (SWIR1, NIR, BLUE),
    "Short-wave infrared (B12/B11/B04)": (SWIR2, SWIR1, RED),
}


def load_model(config_path: str, ckpt_path: str) -> torch.nn.Module:
    """Build the model a config describes and load its checkpoint on CPU."""
    config = read_yaml(config_path)
    model = build_scratch_from_config(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _feature_stack(model: torch.nn.Module) -> torch.nn.Sequential:
    """The conv stack, unwrapping the projection wrapper when present."""
    backbone = model.backbone if isinstance(model, VirtualSensorNet) else model
    return backbone.features


def n_blocks(model: torch.nn.Module) -> int:
    """Number of conv-bn-relu-pool blocks in the feature stack."""
    return len(_feature_stack(model)) // 4


@contextmanager
def _relu_not_inplace(model: torch.nn.Module):
    """Backward hooks cannot be attached through an in-place ReLU, so the flag
    is lifted for the duration of the pass and restored afterwards."""
    flipped = [m for m in model.modules()
               if isinstance(m, torch.nn.ReLU) and m.inplace]
    for m in flipped:
        m.inplace = False
    try:
        yield
    finally:
        for m in flipped:
            m.inplace = True


def grad_cam(model: torch.nn.Module, x: torch.Tensor,
             class_idx: Optional[int] = None,
             block: int = -1) -> Tuple[np.ndarray, int, np.ndarray]:
    """Grad-CAM over one sample.

    x is [1, C, H, W] already normalised. Returns the [h, w] map in [0, 1],
    the class it explains, and the full softmax vector.
    """
    feats = _feature_stack(model)
    total = len(feats) // 4
    b = block if block >= 0 else total + block
    target = feats[4 * b + 2]  # ReLU of that block

    store: Dict[str, torch.Tensor] = {}
    with _relu_not_inplace(model):
        h_fwd = target.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("act", out))
        h_bwd = target.register_full_backward_hook(
            lambda _m, _gi, gout: store.__setitem__("grad", gout[0]))
        try:
            model.zero_grad(set_to_none=True)
            logits = model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

    act, grad = store["act"][0], store["grad"][0]        # [C, h, w]
    weights = grad.mean(dim=(1, 2), keepdim=True)        # channel importance
    cam = torch.relu((weights * act).sum(dim=0)).detach()
    cam = cam - cam.min()
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    probs = torch.softmax(logits, dim=1)[0].detach().numpy()
    return cam.numpy(), int(class_idx), probs


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upsample of a small CAM to the patch size."""
    t = torch.from_numpy(cam)[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                         align_corners=False)
    return up[0, 0].numpy()


def composite(img13: torch.Tensor, combo: str, lo: float = 2.0,
              hi: float = 98.0) -> np.ndarray:
    """Render three bands of a raw [13, H, W] patch as a display image.

    Each channel is independently percentile-stretched, which is the usual way
    Sentinel-2 reflectance is made viewable; it is a display choice only and
    does not affect anything the models see.
    """
    idx = COMPOSITES[combo]
    chans = []
    for b in idx:
        c = img13[b].numpy().astype(np.float32)
        p_lo, p_hi = np.percentile(c, lo), np.percentile(c, hi)
        c = np.clip((c - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
        chans.append(c)
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)


def sample_indices(root: str, split: str, per_class: int,
                   seed: int = 0) -> Dict[int, List[int]]:
    """Pick up to `per_class` dataset indices for each label."""
    base = _base_dataset(root, split)
    targets = np.asarray(base.targets) if hasattr(base, "targets") else None
    picked: Dict[int, List[int]] = {}
    rng = np.random.default_rng(seed)
    if targets is not None:
        for lab in np.unique(targets):
            hits = np.flatnonzero(targets == lab)
            take = rng.choice(hits, size=min(per_class, len(hits)), replace=False)
            picked[int(lab)] = sorted(int(i) for i in take)
        return picked
    order = rng.permutation(len(base))
    for i in order:
        lab = int(base[int(i)]["label"])
        got = picked.setdefault(lab, [])
        if len(got) < per_class:
            got.append(int(i))
        if len(picked) == 10 and all(len(v) >= per_class for v in picked.values()):
            break
    return picked


def class_spectra(root: str, split: str = "test", per_class: int = 24,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of raw reflectance per class per band.

    Returns two [10, 13] arrays indexed by label and by raw band position.
    """
    base = _base_dataset(root, split)
    picked = sample_indices(root, split, per_class, seed)
    means = np.zeros((10, 13), dtype=np.float64)
    sds = np.zeros((10, 13), dtype=np.float64)
    for lab, idxs in picked.items():
        stack = np.stack([base[i]["image"].float().mean(dim=(1, 2)).numpy()
                          for i in idxs])
        means[lab] = stack.mean(axis=0)
        sds[lab] = stack.std(axis=0)
    return means, sds


# Central wavelength in nm, used to order bands physically rather than by the
# array position they happen to occupy in the GeoTIFF.
WAVELENGTH_NM = {"B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705,
                 "B06": 740, "B07": 783, "B08": 842, "B8A": 865, "B09": 945,
                 "B11": 1610, "B12": 2190}

# Position of each band in the raw 13-band stack.
RAW_INDEX = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5,
             "B07": 6, "B08": 7, "B09": 8, "B10": 9, "B11": 10, "B12": 11,
             "B8A": 12}


def make_dataset(config_path: str, split: str = "test"):
    """The dataset a config describes, so inputs match training exactly."""
    from src.data.eurosat_ms import EuroSATMS, compute_or_load_stats

    data = read_yaml(config_path)["data"]
    root = data["root_ms"]
    if not Path(root).is_absolute():
        # Configs store a repo-relative path; resolve it against the repo so the
        # loader does not depend on the caller's working directory.
        root = str(Path(__file__).resolve().parents[2] / root)
    subset = data.get("band_subset", "all")
    drop_cirrus = data.get("drop_cirrus", True)
    add_indices = data.get("add_indices", False)
    stats = compute_or_load_stats(root, subset, drop_cirrus, add_indices)
    return EuroSATMS(root, split, stats, band_subset=subset,
                     drop_cirrus=drop_cirrus, add_indices=add_indices,
                     augment=False)
