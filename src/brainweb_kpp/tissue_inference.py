"""Three-rate synthetic inference on the full BrainWeb tissue domain.

The observation region is fixed in world coordinates before data generation.
Tissue labels are known; CSF/extracerebral tissue have no cell transport or growth.
This module leaves the published homogeneous recovery implementation unchanged.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from scipy.stats import norm

from .model import advance


@dataclass(frozen=True)
class TissueRecoveryDesign:
    dt_days: float = 0.5
    seed_center_mm: tuple = (-30.0, 25.0, 30.0)
    seed_sigma_mm: tuple = (6.0, 6.0, 6.0)
    seed_peak: float = 0.2
    observation_center_mm: tuple = (-30.0, 24.0, 30.0)
    observation_voxels: int = 10
    voxel_mm: float = 6.0
    blur_sigma_mm: float = 2.0
    quadrature_points: int = 3


@dataclass
class TissueFit:
    parameters: np.ndarray                # D_WM, D_GM [mm²/day], rho [1/day]
    log_covariance: np.ndarray
    intervals: np.ndarray                 # marginal local confidence intervals
    correlation: np.ndarray              # full 3 x 3 log-rate correlation
    singular_values: np.ndarray
    standardized_residuals: np.ndarray
    at_bound: bool
    evaluations: int


class BrainWebExperiment:
    """No-flux heterogeneous Fisher–KPP using the explorer's float64 kernel.

    Growth is shared across WM (label 3), GM (2) and glial tissue (8). Glial
    tissue uses the GM diffusion rate. The imaging signal is blurred normalized
    density, averaged by midpoint quadrature inside fixed physical voxels.
    """

    parameter_names = ("D_WM", "D_GM", "rho")

    def __init__(self, brain, design=TissueRecoveryDesign()):
        self.brain, self.design = brain, design
        h = float(brain.spacing)
        if not np.isfinite(h) or h <= 0 or not np.allclose(brain.affine[:3, :3], np.eye(3) * h):
            raise ValueError("Need an axis-aligned, positive RAS grid")
        for name in ("seed_center_mm", "seed_sigma_mm", "observation_center_mm"):
            value = np.asarray(getattr(design, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
        if min(design.seed_sigma_mm) <= 0 or not np.isfinite(design.seed_peak) or not 0 < design.seed_peak <= 1:
            raise ValueError("Need positive seed widths and a peak in (0, 1]")
        for name in ("dt_days", "voxel_mm", "blur_sigma_mm"):
            value = getattr(design, name)
            if not np.isfinite(value) or value < 0 or (name != "blur_sigma_mm" and value == 0):
                raise ValueError(f"Invalid {name}")
        if any(not isinstance(getattr(design, name), int) or getattr(design, name) < 1
               for name in ("observation_voxels", "quadrature_points")):
            raise ValueError("Observation and quadrature counts must be positive integers")
        self.shape = brain.labels.shape
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("Need a three-dimensional tissue volume")
        self.mask = np.isin(brain.labels, [2, 3, 8])
        if not self.mask[brain.voxel(design.seed_center_mm)]:
            raise ValueError("Seed center must be in parenchyma")
        self.active = np.flatnonzero(self.mask.ravel())
        self.strides = np.array([self.shape[1] * self.shape[2], self.shape[2], 1])
        self.capacity = np.ones(self.mask.size)
        r2 = np.zeros(self.shape)
        for axis in range(3):
            coord = brain.affine[axis, 3] + np.arange(self.shape[axis]) * h
            shape = [1, 1, 1]
            shape[axis] = len(coord)
            r2 += ((coord - design.seed_center_mm[axis]) / design.seed_sigma_mm[axis]).reshape(shape)**2
        self.initial = np.ascontiguousarray(design.seed_peak * np.exp(-r2 / 2) * self.mask)
        # Fixed face types let each fit evaluate the exact harmonic WM/GM law
        # with a small lookup, without recomputing anatomy or wrapping boundaries.
        self.face_types = np.zeros((3, *self.shape), dtype=np.uint8)
        white = brain.labels == 3
        for axis in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            both = self.mask[lo] & self.mask[hi]
            types = np.where(white[lo] & white[hi], 1,
                             np.where(~white[lo] & ~white[hi], 2, 3))
            self.face_types[axis][lo] = np.where(both, types, 0)
        self.face_types = np.ascontiguousarray(self.face_types.reshape(3, -1))

        m, q = design.observation_voxels, design.quadrature_points
        offsets = (np.arange(m) + .5 - m / 2) * design.voxel_mm
        self.observation_axes = [center + offsets for center in design.observation_center_mm]
        centers = np.stack(np.meshgrid(*self.observation_axes, indexing="ij"), axis=-1)
        sub = (np.arange(q) + .5 - q / 2) * design.voxel_mm / q
        quadrature = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), axis=-1).reshape(-1, 3)
        world = centers.reshape(-1, 1, 3) + quadrature[None, :, :]
        self.sample_indices = np.moveaxis((world - brain.affine[:3, 3]) / h, -1, 0)
        if np.any(self.sample_indices < 0) or any(
                np.any(self.sample_indices[axis] > self.shape[axis] - 1) for axis in range(3)):
            raise ValueError("Observation voxels must lie inside the image")
        self.observation_shape = (m, m, m)

    def faces(self, d_white, d_gray):
        if not np.isfinite(d_white) or not np.isfinite(d_gray) or min(d_white, d_gray) < 0:
            raise ValueError("Diffusion rates must be finite and nonnegative")
        harmonic = 2 * d_white * d_gray / (d_white + d_gray) if d_white + d_gray else 0
        lookup = np.array([0, d_white, d_gray, harmonic]) / self.brain.spacing**2
        return np.ascontiguousarray(lookup[self.face_types])

    def observe(self, volume):
        if volume.shape != self.shape:
            raise ValueError("Field does not match the tissue grid")
        smooth = gaussian_filter(volume, self.design.blur_sigma_mm / self.brain.spacing,
                                 mode="constant", cval=0.)
        samples = map_coordinates(smooth, self.sample_indices, order=1,
                                  mode="constant", cval=0., prefilter=False)
        return samples.mean(axis=-1).reshape(self.observation_shape)

    def _evolve(self, d_white, d_gray, rho, times, store):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, nonnegative and strictly increasing")
        faces = self.faces(d_white, d_gray)
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("Proliferation must be finite and nonnegative")
        if 6 * max(d_white, d_gray) * self.design.dt_days / self.brain.spacing**2 > .9 + 1e-12:
            raise ValueError("Requested time step violates the diffusion stability limit")
        n, out = self.initial.ravel().copy(), np.zeros(self.mask.size)
        result, previous = [], 0.
        for target in times:
            steps = max(1, math.ceil((target - previous) / self.design.dt_days))
            dt = (target - previous) / steps
            decay = np.full(n.size, np.exp(-rho * dt))
            for _ in range(steps):
                advance(n, out, faces, self.active, self.strides, decay, self.capacity, dt)
                n, out = out, n
            field = n.reshape(self.shape)
            result.append(field.copy() if store else self.observe(field))
            previous = target
        return np.asarray(result)

    def predict(self, d_white, d_gray, rho, times):
        return self._evolve(d_white, d_gray, rho, times, store=False)

    def states(self, d_white, d_gray, rho, times):
        return self._evolve(d_white, d_gray, rho, times, store=True)


def fit_tissue_parameters(experiment, times, observations, noise_sd, *,
                          start=(.4, .15, .035),
                          bounds=((.05, .02, .01), (1., .6, .09)), confidence=.95):
    """Three-rate Gaussian MLE, with local known-noise confidence intervals.

    Bounds define a positive search region; the WM:GM ratio is not fixed or ordered.
    Near-bound fits are flagged. Rank deficiency is rejected. No prior is used.
    """
    if not np.isfinite(noise_sd) or noise_sd <= 0 or not 0 < confidence < 1:
        raise ValueError("Need positive noise SD and confidence in (0, 1)")
    start, bounds = np.asarray(start, float), np.asarray(bounds, float)
    if start.shape != (3,) or bounds.shape != (2, 3) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(bounds)) or np.any(bounds <= 0) or np.any(bounds[0] >= bounds[1]) or np.any(start <= bounds[0]) or np.any(start >= bounds[1]):
        raise ValueError("Need three positive rates and strictly enclosing bounds")
    observations = np.asarray(observations, float)
    if observations.shape != (len(times), *experiment.observation_shape) or not np.all(np.isfinite(observations)):
        raise ValueError("Invalid observation shape or nonfinite values")

    def residual(log_rates):
        return ((experiment.predict(*np.exp(log_rates), times) - observations) / noise_sd).ravel()

    fit = least_squares(residual, np.log(start), bounds=np.log(bounds), jac="3-point",
                        diff_step=1e-4, x_scale="jac", ftol=1e-8, xtol=1e-8, gtol=1e-6)
    if not fit.success:
        raise RuntimeError(f"Tissue fit failed: {fit.message}")
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    if len(singular) != 3 or singular[-1] <= singular[0] * 1e-8:
        raise ValueError("WM diffusion, GM diffusion and proliferation are not locally identifiable")
    covariance = (vt.T / singular**2) @ vt
    sd = np.sqrt(np.diag(covariance))
    intervals = np.exp(fit.x[:, None] + norm.ppf(.5 + confidence / 2) * sd[:, None] * [-1, 1])
    at_bound = np.any(np.minimum(fit.x - np.log(bounds[0]), np.log(bounds[1]) - fit.x) < 1e-4)
    return TissueFit(np.exp(fit.x), covariance, intervals, covariance / np.outer(sd, sd),
                     singular, fit.fun, bool(at_bound), fit.nfev)
