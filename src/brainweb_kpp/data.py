"""BrainWeb I/O. All simulation arrays use canonical (x, y, z), RAS+ axes."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
T1_NAME = "t1_icbm_normal_1mm_pn0_rf0.nii.gz"
LABEL_NAME = "phantom_1.0mm_normal_crisp.nii.gz"


@dataclass
class Brain:
    t1: np.ndarray
    labels: np.ndarray
    affine: np.ndarray
    spacing: float

    @property
    def mask(self):
        # CSF may transport cells, but does not proliferate. Glial class -> GM.
        return np.isin(self.labels, [1, 2, 3, 8])

    def voxel(self, xyz):
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(self.affine), xyz)).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= self.labels.shape):
            raise ValueError("Origin is outside the image. Use the coordinate sliders.")
        return tuple(ijk)

    def world(self, ijk):
        return nib.affines.apply_affine(self.affine, ijk)


def verify_data(data_dir=DATA):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run uv run python scripts/fetch_data.py")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}; restore the bundled data.")
    return manifest


def load_brainweb(spacing=2.0, data_dir=DATA):
    """Resample T1 linearly and ground-truth labels by nearest neighbor.

    A spacing below 1 mm interpolates anatomy; it creates no new MRI information.
    The lower RAS coordinate remains fixed; the upper edge may move by < spacing.
    """
    spacing = float(spacing)
    if not np.isfinite(spacing) or not 0.5 <= spacing <= 4:
        raise ValueError("Spatial step must be between 0.5 and 4 mm.")
    data_dir = Path(data_dir)
    t1 = nib.as_closest_canonical(nib.load(data_dir / T1_NAME))
    labels = nib.as_closest_canonical(nib.load(data_dir / LABEL_NAME))
    if t1.shape != labels.shape or not np.allclose(t1.affine, labels.affine, atol=1e-4):
        raise ValueError("BrainWeb T1 and labels are not aligned.")
    if not np.allclose(t1.header.get_zooms()[:3], 1.0):
        raise ValueError("Expected the native 1 mm BrainWeb volume.")
    affine = t1.affine.copy()
    if spacing != 1.0:
        shape = tuple((np.floor((np.array(t1.shape) - 1) / spacing) + 1).astype(int))
        affine[:3, :3] = np.eye(3) * spacing
        t1 = resample_from_to(t1, (shape, affine), order=1)
        labels = resample_from_to(labels, (shape, affine), order=0)
    image = t1.get_fdata(dtype=np.float32)
    image /= max(float(np.percentile(image, 99.8)), 1e-8)
    return Brain(np.ascontiguousarray(image),
                 np.ascontiguousarray(np.asarray(labels.dataobj), dtype=np.uint8),
                 affine, spacing)
