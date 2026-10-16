"""Conservative cell-centered fluxes + exact logistic reaction (Lie splitting).

No periodic wraparound, absorbing-mask multiplication, or density clipping.
"""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


@dataclass(frozen=True)
class Parameters:
    origin_mm: tuple = (-30.0, 25.0, 30.0)
    radii_mm: tuple = (5.0, 5.0, 5.0)
    seed_profile: str = "gaussian"
    initial_density: float = 20000.0  # cells/mm³; Gaussian peak or uniform ellipsoid
    d_white: float = 0.5             # mm²/day
    d_gray: float = 0.1
    d_csf: float = 0.0
    spatial_multiplier: float = 1.0  # D multiplier at the origin; tends to 1 remotely
    spatial_width_mm: float = 30.0
    rho: float = 0.04               # day⁻¹; zero in CSF
    carrying_capacity: float = 100000.0  # cells/mm³
    duration_days: float = 180.0
    dt_days: float = 0.25           # requested maximum internal step
    frame_days: float = 3.0         # output cadence, independent of numerical step
    auto_substep: bool = True

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if len(self.origin_mm) != 3 or not np.all(np.isfinite(self.origin_mm)):
            raise ValueError("Origin must contain three finite RAS coordinates")
        if len(self.radii_mm) != 3 or not np.all(np.isfinite(self.radii_mm)) or min(self.radii_mm) <= 0:
            raise ValueError("Seed radii must contain three positive finite lengths")
        if min(self.d_white, self.d_gray, self.d_csf, self.rho) < 0:
            raise ValueError("Diffusion and proliferation must be nonnegative")
        if min(self.dt_days, self.frame_days, self.spatial_width_mm, self.spatial_multiplier, self.carrying_capacity) <= 0:
            raise ValueError("Time steps, capacity, width and multiplier must be positive")
        if self.duration_days <= 0:
            raise ValueError("Duration must be positive")
        if not 0 < self.initial_density <= self.carrying_capacity:
            raise ValueError("Initial density must lie in (0, carrying capacity]")
        if self.seed_profile not in ("gaussian", "ellipsoid"):
            raise ValueError("Seed profile must be gaussian or ellipsoid")


def fields(brain, p):
    p.validate()
    lab = brain.labels
    d = np.zeros(lab.shape, dtype=np.float32)
    d[lab == 3] = p.d_white
    d[np.isin(lab, [2, 8])] = p.d_gray
    d[lab == 1] = p.d_csf
    if p.spatial_multiplier != 1:
        r2 = distance_squared(brain, p.origin_mm, (p.spatial_width_mm,) * 3)
        d *= 1 + (p.spatial_multiplier - 1) * np.exp(-r2 / 2)
    rho = np.where(np.isin(lab, [2, 3, 8]), p.rho, 0).astype(np.float32)
    capacity = np.full(lab.shape, p.carrying_capacity, dtype=np.float32)
    return d, rho, capacity


def distance_squared(brain, origin, radii):
    out = np.zeros(brain.labels.shape, dtype=np.float32)
    for axis, (center, radius) in enumerate(zip(origin, radii)):
        coord = brain.affine[axis, 3] + np.arange(out.shape[axis], dtype=np.float32) * brain.spacing
        shape = [1, 1, 1]
        shape[axis] = len(coord)
        out += ((coord - center) / radius).reshape(shape) ** 2
    return out


def initial_condition(brain, p):
    p.validate()
    ijk = brain.voxel(p.origin_mm)
    if brain.labels[ijk] not in (2, 3, 8):
        raise ValueError("Place the seed center in white/gray matter (use the tissue preview).")
    r2 = distance_squared(brain, p.origin_mm, p.radii_mm)
    if p.seed_profile == "gaussian":
        # Explicit compact truncation avoids tiny Gaussian tails everywhere in brain.
        n = np.where(r2 <= 9, p.initial_density * np.exp(-r2 / 2), 0)
    else:
        n = np.where(r2 <= 1, p.initial_density, 0)
    n[~np.isin(brain.labels, [2, 3, 8])] = 0
    return np.ascontiguousarray(n, dtype=np.float32)


def face_coefficients(d, mask, spacing):
    """Positive-face D/h². Harmonic mean gives exactly zero across barriers."""
    faces = []
    total = np.zeros_like(d)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        a, b = d[lo], d[hi]
        f = np.zeros_like(d)
        np.divide(2 * a * b, a + b, out=f[lo], where=(a + b) > 0)
        f[lo] *= mask[lo] & mask[hi]
        f /= spacing ** 2
        total[lo] += f[lo]
        total[hi] += f[lo]
        faces.append(f.ravel())
    rate = float(total.max())
    return np.ascontiguousarray(faces), (0.9 / rate if rate > 0 else math.inf)


@njit(cache=True, nogil=True, parallel=True)
def advance(n, out, faces, active, strides, decay, capacity, dt):
    for q in prange(active.size):
        i = active[q]
        value = n[i]
        flux = 0.0
        for axis in range(3):
            s = strides[axis]
            # Zero face coefficients also handle outside-volume boundary faces.
            fp = faces[axis, i]
            if fp > 0:
                flux += fp * (n[i + s] - value)
            if i >= s:
                fm = faces[axis, i - s]
                if fm > 0:
                    flux += fm * (n[i - s] - value)
        diffused = value + dt * flux
        # Exact nonnegative logistic flow, algebraically stable even at n = 0.
        out[i] = diffused / (decay[i] + (diffused / capacity[i]) * (1 - decay[i]))


@dataclass
class Result:
    times: np.ndarray
    planes: dict
    backgrounds: dict
    final_density: np.ndarray
    metrics: dict
    parameters: dict
    metadata: dict
    affine: np.ndarray
    indices: tuple
    volumes: object = None


def get_planes(volume, indices):
    x, y, z = indices
    return {"Axial": volume[:, :, z].copy(), "Coronal": volume[:, y, :].copy(),
            "Sagittal": volume[x, :, :].copy()}


def simulate(brain, p=Parameters(), *, store_volumes=False, progress=None,
             diffusion=None, proliferation=None, capacity=None):
    """3-D simulation; optional custom coefficient arrays are on brain's grid.

    Custom spatial capacity can be exceeded by diffusion locally; no clipping is
    applied. Notebook controls use uniform capacity. Snapshot volumes use float32.
    """
    start = time.perf_counter()
    d, rho, k = fields(brain, p)
    for name, supplied in (("diffusion", diffusion), ("proliferation", proliferation), ("capacity", capacity)):
        if supplied is None:
            continue
        a = np.asarray(supplied, dtype=np.float32)
        if a.shape != brain.labels.shape or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"Invalid {name} map: require finite nonnegative values on the MRI grid")
        if name == "capacity" and np.any(a <= 0):
            raise ValueError("Capacity map must be strictly positive")
        if name == "diffusion": d = a.copy()
        elif name == "proliferation": rho = a.copy()
        else: k = a.copy()
    mask = brain.mask
    d[~mask] = 0
    rho[~np.isin(brain.labels, [2, 3, 8])] = 0
    faces, limit = face_coefficients(d, mask, brain.spacing)
    if p.dt_days > limit and not p.auto_substep:
        raise ValueError(f"Requested dt={p.dt_days:g} exceeds safe diffusion step {limit:.6g} days")
    dt_max = min(p.dt_days, limit)
    times = np.r_[np.arange(0, p.duration_days, p.frame_days), p.duration_days]
    if len(times) > 2001:
        raise ValueError("Too many frames (>2001). Increase frame_days.")
    if store_volumes and len(times) * mask.size * 4 > 8e9:
        raise ValueError("Snapshot storage exceeds 8 GB. Increase frame_days or spatial step.")
    n = initial_condition(brain, p).ravel()
    out = np.zeros_like(n)
    active = np.flatnonzero(mask.ravel()).astype(np.int32)
    strides = np.array([mask.shape[1] * mask.shape[2], mask.shape[2], 1])
    indices = brain.voxel(p.origin_mm)
    planes = {name: [] for name in get_planes(brain.t1, indices)}
    volumes = [] if store_volumes else None
    metrics = {"cells": [], "volume_20_ml": [], "volume_80_ml": [], "max_density": []}
    steps, actual_max = 0, 0.0
    # Limit the default thread count to avoid oversubscribing notebook/BLAS work.
    set_num_threads(min(get_num_threads(), 8))
    for frame, target in enumerate(times):
        if frame:
            interval = target - times[frame - 1]
            count = max(1, int(np.ceil(interval / dt_max)))
            dt = interval / count  # land exactly on every output time
            decay = np.exp(-rho.ravel() * dt).astype(np.float32)
            for _ in range(count):
                advance(n, out, faces, active, strides, decay, k.ravel(), dt)
                n, out = out, n
            steps += count
            actual_max = max(actual_max, dt)
        volume = n.reshape(mask.shape)
        if not np.all(np.isfinite(volume)) or volume.min() < -1e-4:
            raise FloatingPointError("Nonfinite/negative density: inspect coefficients and time step")
        for name, plane in get_planes(volume, indices).items():
            planes[name].append(plane)
        if store_volumes:
            volumes.append(volume.copy())
        metrics["cells"].append(float(volume.sum(dtype=np.float64) * brain.spacing ** 3))
        for threshold, name in ((0.2, "volume_20_ml"), (0.8, "volume_80_ml")):
            metrics[name].append(float(np.count_nonzero((volume >= threshold * k) & mask)
                                       * brain.spacing ** 3 / 1000))
        metrics["max_density"].append(float(volume.max()))
        if progress:
            progress(frame + 1, len(times))
    return Result(times, {k: np.asarray(v) for k, v in planes.items()},
                  get_planes(brain.t1, indices), n.reshape(mask.shape).copy(),
                  {k: np.asarray(v) for k, v in metrics.items()}, asdict(p),
                  {"spacing_mm": brain.spacing, "shape": list(mask.shape),
                   "safe_dt_days": limit if np.isfinite(limit) else None,
                   "actual_max_dt_days": actual_max, "internal_steps": steps,
                   "elapsed_seconds": time.perf_counter() - start,
                   "coefficient_maps": {"diffusion": diffusion is not None,
                                        "proliferation": proliferation is not None,
                                        "capacity": capacity is not None}},
                  brain.affine.copy(), indices, volumes)
