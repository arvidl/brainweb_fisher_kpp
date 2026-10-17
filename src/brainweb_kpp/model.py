"""Conservative cell-centered fluxes + exact logistic reaction (Lie splitting).

No periodic wraparound, absorbing-mask multiplication, or density clipping.
"""
from dataclasses import dataclass, asdict, replace
import math
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.sparse.linalg import LinearOperator, cg


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
    model: str = "fisher_kpp"
    oxygen_length_mm: float = 0.2
    oxygen_uptake_ratio: float = 4.0
    perfusion_suppression: float = 4.0
    oxygen_prolif_level: float = 0.3
    oxygen_death_level: float = 0.1
    necrosis_rate: float = 0.5       # day⁻¹
    necrosis_clearance: float = 0.0  # day⁻¹
    oxygen_rtol: float = 1e-7
    oxygen_maxiter: int = 200

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
        if self.model not in ("fisher_kpp", "coupled"):
            raise ValueError("Model must be fisher_kpp or coupled")
        if self.model == "fisher_kpp":
            return  # Inactive coupled controls must not prevent a baseline run.
        if min(self.oxygen_length_mm, self.oxygen_uptake_ratio, self.perfusion_suppression,
               self.necrosis_rate, self.necrosis_clearance) < 0:
            raise ValueError("Oxygen, perfusion and necrosis coefficients must be nonnegative")
        if not 0 < self.oxygen_death_level < self.oxygen_prolif_level < 1:
            raise ValueError("Oxygen thresholds must satisfy 0 < death < proliferation < 1")
        if not 0 < self.oxygen_rtol < 1:
            raise ValueError("Oxygen solver relative tolerance must lie between 0 and 1")
        if not isinstance(self.oxygen_maxiter, int) or self.oxygen_maxiter < 1:
            raise ValueError("Oxygen solver iteration limit must be a positive integer")


def fields(brain, p):
    p.validate()
    lab = brain.labels
    d = np.zeros(lab.shape, dtype=np.float32)
    d[lab == 3] = p.d_white
    d[np.isin(lab, [2, 8])] = p.d_gray
    d[lab == 1] = p.d_csf if p.model == "fisher_kpp" else 0
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


@njit(cache=True, nogil=True, parallel=True)
def coupled_reaction(c, z, oxygen, rho, capacity, active, dt, theta_p, theta_d,
                     death_rate, clearance):
    """Exact frozen-oxygen birth, conservative death transfer, then clearance."""
    for q in prange(active.size):
        i = active[q]
        free = capacity[i] - z[i]
        if free > 0 and c[i] < free:
            rate = rho[i] * min(1., oxygen[i] / theta_p) * free / capacity[i]
            decay = math.exp(-rate * dt)
            c[i] = c[i] / (decay + (c[i] / free) * (1 - decay))
        rate = death_rate * max(1 - oxygen[i] / theta_d, 0.)
        lost = c[i] * (-math.expm1(-rate * dt))
        c[i] -= lost
        z[i] = (z[i] + lost) * math.exp(-clearance * dt)


@njit(cache=True, nogil=True, parallel=True)
def oxygen_matvec(x, reaction, neighbors, weight):
    out = np.empty_like(x)
    for i in prange(x.size):
        value = reaction[i] * x[i]
        for j in range(6):
            other = neighbors[i, j]
            if other >= 0:
                value += weight * (x[i] - x[other])
        out[i] = value
    return out


class OxygenSolver:
    """Compact parenchymal no-flux stencil, matrix-free preconditioned CG.

    Only the reaction diagonal changes. Float64 CG avoids loss of accuracy in
    the elliptic solve; stored oxygen uses float32, like the cell fields.
    """
    def __init__(self, mask, spacing, p):
        self.mask, self.p = mask, p
        self.active = np.flatnonzero(mask.ravel())
        if not self.active.size:
            raise ValueError("The coupled model requires gray/white/glial tissue")
        mapping = np.full(mask.shape, -1, np.int32)
        mapping.ravel()[self.active] = np.arange(self.active.size, dtype=np.int32)
        self.neighbors = np.full((self.active.size, 6), -1, np.int32)
        for axis in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            a, b = mapping[tuple(lo)].ravel(), mapping[tuple(hi)].ravel()
            valid = (a >= 0) & (b >= 0)
            self.neighbors[a[valid], 2 * axis] = b[valid]
            self.neighbors[b[valid], 2 * axis + 1] = a[valid]
        self.weight = (p.oxygen_length_mm / spacing) ** 2
        self.degree = np.count_nonzero(self.neighbors >= 0, axis=1)
        self.previous = np.ones(self.active.size)
        self.solves, self.iterations, self.max_residual = 0, 0, 0.

    def solve(self, c, z, capacity):
        ids, p = self.active, self.p
        u = c.ravel()[ids].astype(np.float64) / capacity.ravel()[ids]
        burden = u + z.ravel()[ids].astype(np.float64) / capacity.ravel()[ids]
        supply = 1 / (1 + p.perfusion_suppression * burden)
        reaction = supply + p.oxygen_uptake_ratio * u
        diagonal = reaction + self.weight * self.degree
        iterations = 0
        def counted(x):
            nonlocal iterations
            iterations += 1
        if p.oxygen_uptake_ratio == 0 or not np.any(u):
            solution, info, residual = np.ones(ids.size), 0, 0.
        elif self.weight == 0:
            solution, info, residual = supply / reaction, 0, 0.
        else:
            operator = LinearOperator((ids.size, ids.size), dtype=np.float64,
                                      matvec=lambda x: oxygen_matvec(x, reaction, self.neighbors, self.weight))
            preconditioner = LinearOperator(operator.shape, dtype=np.float64, matvec=lambda x: x / diagonal)
            solution, info = cg(operator, supply, x0=self.previous, M=preconditioner,
                                rtol=p.oxygen_rtol, atol=0., maxiter=p.oxygen_maxiter, callback=counted)
            residual = np.linalg.norm(operator @ solution - supply) / np.linalg.norm(supply)
        if info != 0 or not np.isfinite(residual) or residual > p.oxygen_rtol * 1.1:
            raise RuntimeError(f"Oxygen solve did not converge (status={info}, relative residual={residual:.3g})")
        if not np.all(np.isfinite(solution)) or solution.min() <= 0 or solution.max() > 1 + 1e-6:
            raise FloatingPointError("Oxygen solve violated 0 < oxygen ≤ 1; reduce solver tolerance")
        self.previous = solution
        self.solves += 1
        self.iterations += iterations
        self.max_residual = max(self.max_residual, residual)
        out = np.zeros(self.mask.shape, np.float32)
        out.ravel()[ids] = solution
        return out


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
    final_oxygen: object = None
    final_necrosis: object = None
    oxygen_planes: object = None
    necrosis_planes: object = None
    oxygen_volumes: object = None
    necrosis_volumes: object = None
    tissue_mask: object = None
    comparison: object = None


def get_planes(volume, indices):
    x, y, z = indices
    return {"Axial": volume[:, :, z].copy(), "Coronal": volume[:, y, :].copy(),
            "Sagittal": volume[x, :, :].copy()}


def simulate(brain, p=Parameters(), *, store_volumes=False, progress=None,
             diffusion=None, proliferation=None, capacity=None, compare=False):
    """3-D simulation; optional custom coefficient arrays are on brain's grid.

    Custom spatial capacity can be exceeded by diffusion locally; no clipping is
    applied. Notebook controls use uniform capacity. Snapshot volumes use float32.
    """
    start = time.perf_counter()
    if compare and p.model != "coupled":
        raise ValueError("A Fisher–KPP reference is available when model='coupled'")
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
    coupled = p.model == "coupled"
    mask = np.isin(brain.labels, [2, 3, 8]) if coupled else brain.mask
    d[~mask] = 0
    rho[~np.isin(brain.labels, [2, 3, 8])] = 0
    faces, limit = face_coefficients(d, mask, brain.spacing)
    if p.dt_days > limit and not p.auto_substep:
        raise ValueError(f"Requested dt={p.dt_days:g} exceeds safe diffusion step {limit:.6g} days")
    dt_max = min(p.dt_days, limit)
    times = np.r_[np.arange(0, p.duration_days, p.frame_days), p.duration_days]
    if len(times) > 2001:
        raise ValueError("Too many frames (>2001). Increase frame_days.")
    field_count = (3 if coupled else 1) + int(compare)
    if store_volumes and len(times) * mask.size * 4 * field_count > 8e9:
        raise ValueError("Snapshot storage exceeds 8 GB. Increase frame_days or spatial step.")
    n = initial_condition(brain, p).ravel()
    out = np.zeros_like(n)
    active = np.flatnonzero(mask.ravel()).astype(np.int32)
    strides = np.array([mask.shape[1] * mask.shape[2], mask.shape[2], 1])
    indices = brain.voxel(p.origin_mm)
    planes = {name: [] for name in get_planes(brain.t1, indices)}
    volumes = [] if store_volumes else None
    metrics = {"cells": [], "volume_20_ml": [], "volume_80_ml": [], "max_density": []}
    oxygen_planes = {name: [] for name in planes} if coupled else None
    necrosis_planes = {name: [] for name in planes} if coupled else None
    oxygen_volumes, necrosis_volumes = [], []
    z = np.zeros_like(n) if coupled else None
    diffusion_decay = np.ones_like(n) if coupled else None
    steps, actual_max = 0, 0.0
    # Limit the default thread count to avoid oversubscribing notebook/BLAS work.
    set_num_threads(min(get_num_threads(), 8))
    oxygen_solver = OxygenSolver(mask, brain.spacing, p) if coupled else None
    oxygen = oxygen_solver.solve(n, z, k) if coupled else None
    if coupled:
        metrics.update(necrotic_cells=[], necrotic_volume_20_ml=[], hypoxic_volume_ml=[], min_oxygen=[])
    for frame, target in enumerate(times):
        if frame:
            interval = target - times[frame - 1]
            count = max(1, int(np.ceil(interval / dt_max)))
            dt = interval / count  # land exactly on every output time
            decay = np.exp(-rho.ravel() * dt).astype(np.float32)
            for _ in range(count):
                advance(n, out, faces, active, strides, diffusion_decay if coupled else decay, k.ravel(), dt)
                n, out = out, n
                if coupled:
                    coupled_reaction(n, z, oxygen.ravel(), rho.ravel(), k.ravel(), active, dt,
                                     p.oxygen_prolif_level, p.oxygen_death_level,
                                     p.necrosis_rate, p.necrosis_clearance)
                    oxygen = oxygen_solver.solve(n, z, k)
            steps += count
            actual_max = max(actual_max, dt)
        volume = n.reshape(mask.shape)
        if not np.all(np.isfinite(volume)) or volume.min() < -1e-4:
            raise FloatingPointError("Nonfinite/negative density: inspect coefficients and time step")
        for name, plane in get_planes(volume, indices).items():
            planes[name].append(plane)
        if store_volumes:
            volumes.append(volume.copy())
        if coupled:
            dead = z.reshape(mask.shape)
            if not np.all(np.isfinite(dead)) or dead.min() < 0:
                raise FloatingPointError("Nonfinite/negative necrotic density")
            for name, plane in get_planes(oxygen, indices).items(): oxygen_planes[name].append(plane.copy())
            for name, plane in get_planes(dead, indices).items(): necrosis_planes[name].append(plane.copy())
            if store_volumes:
                oxygen_volumes.append(oxygen.copy())
                necrosis_volumes.append(dead.copy())
            metrics["necrotic_cells"].append(float(dead.sum(dtype=np.float64) * brain.spacing ** 3))
            metrics["necrotic_volume_20_ml"].append(float(np.count_nonzero((dead >= .2 * k) & mask) * brain.spacing ** 3 / 1000))
            metrics["hypoxic_volume_ml"].append(float(np.count_nonzero((oxygen < p.oxygen_death_level) & mask) * brain.spacing ** 3 / 1000))
            metrics["min_oxygen"].append(float(oxygen[mask].min()))
        metrics["cells"].append(float(volume.sum(dtype=np.float64) * brain.spacing ** 3))
        for threshold, name in ((0.2, "volume_20_ml"), (0.8, "volume_80_ml")):
            metrics[name].append(float(np.count_nonzero((volume >= threshold * k) & mask)
                                       * brain.spacing ** 3 / 1000))
        metrics["max_density"].append(float(volume.max()))
        if progress:
            progress(frame + 1, len(times) * (2 if compare else 1))
    result = Result(times, {k: np.asarray(v) for k, v in planes.items()},
                  get_planes(brain.t1, indices), n.reshape(mask.shape).copy(),
                  {k: np.asarray(v) for k, v in metrics.items()}, asdict(p),
                  {"spacing_mm": brain.spacing, "shape": list(mask.shape),
                   "safe_dt_days": limit if np.isfinite(limit) else None,
                   "actual_max_dt_days": actual_max, "internal_steps": steps,
                   "elapsed_seconds": time.perf_counter() - start,
                   "model": p.model,
                   "coefficient_maps": {"diffusion": diffusion is not None,
                                        "proliferation": proliferation is not None,
                                        "capacity": capacity is not None}},
                  brain.affine.copy(), indices, volumes)
    if coupled:
        result.final_oxygen, result.final_necrosis = oxygen, z.reshape(mask.shape).copy()
        result.oxygen_planes = {name: np.asarray(v) for name, v in oxygen_planes.items()}
        result.necrosis_planes = {name: np.asarray(v) for name, v in necrosis_planes.items()}
        result.oxygen_volumes = oxygen_volumes if store_volumes else None
        result.necrosis_volumes = necrosis_volumes if store_volumes else None
        result.tissue_mask = mask
        result.metadata["transport_domain"] = "gray/white/glial tissue; no CSF"
        result.metadata["field_units"] = dict(viable="cells/mm3", necrosis="cell equivalents/mm3",
                                              oxygen="relative to normal tissue; zero outside tissue_mask")
        result.metadata["oxygen_solver"] = dict(solves=oxygen_solver.solves, iterations=oxygen_solver.iterations,
                                                 max_relative_residual=oxygen_solver.max_residual)
    if compare:
        reference_p = replace(p, model="fisher_kpp", d_csf=0.)
        reference_progress = (lambda done, total: progress(total + done, 2 * total)) if progress else None
        # Match the coupled transport domain even with a custom D map or a
        # nonzero CSF control; the reference changes only the reaction model.
        result.comparison = simulate(brain, reference_p, store_volumes=store_volumes, progress=reference_progress,
                                     diffusion=d, proliferation=rho, capacity=k)
        result.comparison.metadata["coefficient_maps"] = result.metadata["coefficient_maps"].copy()
        result.comparison.metadata["transport_domain"] = result.metadata["transport_domain"]
        result.metadata["comparison"] = "reference_fisher_kpp"
        result.metadata["elapsed_seconds_with_reference"] = time.perf_counter() - start
    return result
