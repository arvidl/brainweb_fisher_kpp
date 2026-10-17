"""Conditional synthetic four-parameter recovery on real PAT16 anatomy.

Tissue labels, dMRI directions, initialization and imaging calibration are
treated as known. Synthetic cellularity proxies are not observed patient MRI.
No fixed WM:GM ratio and no Bayesian prior are used.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from scipy.stats import norm

from .anisotropic import migration_tensor, transport_matrix
from .pat16 import initial_density


PARAMETER_NAMES = ("D_WM_mm2_day", "D_GM_mm2_day", "rho_per_day",
                   "parallel_perpendicular")
DEFAULT_BOUNDS = ((.05, .02, .01, 1.), (1., .6, .09, 3.5))


@dataclass(frozen=True)
class PAT16RecoveryDesign:
    dt_days: float = .5
    observation_center_mm: tuple | None = None  # lesion centroid, fixed before fitting
    observation_voxels: int = 12
    voxel_mm: float = 6.
    blur_sigma_mm: float = 2.
    quadrature_points: int = 3
    lesion_threshold: float = .5
    initial_amplitude: float = .6


@dataclass
class PAT16Fit:
    parameters: np.ndarray
    log_covariance: np.ndarray
    intervals: np.ndarray
    correlation: np.ndarray
    singular_values: np.ndarray
    standardized_residuals: np.ndarray
    at_bound: bool
    evaluations: int


@njit(cache=True)
def _states64(indptr, indices, data, initial, times, rho, dt_max):
    """Same diffusion/logistic update as the pilot, with float64 snapshots.

    Float32 snapshots in the display-oriented pilot lose precision when forming
    finite-difference likelihood derivatives. Keep every inference state double.
    """
    u, work = initial.copy(), initial.copy()
    states = np.empty((len(times), len(u)), np.float64)
    time = 0.
    for frame in range(len(times)):
        while time < times[frame] - 1e-12:
            dt = min(dt_max, times[frame] - time)
            for i in range(len(u)):
                rate = 0.
                for edge in range(indptr[i], indptr[i+1]):
                    rate += data[edge] * u[indices[edge]]
                work[i] = u[i] + dt * rate
            factor = np.exp(rho * dt)
            for i in range(len(u)):
                u[i] = factor * work[i] / (1 + (factor - 1) * work[i])
            time += dt
        states[frame] = u
    return states


class PAT16Experiment:
    """Full PAT16 no-flux tissue domain; fixed physical observation cube.

    D_WM and D_GM are independent mean migration diffusivities. Only eligible
    WM receives the trace-matched dMRI tensor with fitted parallel/perpendicular
    ratio. Invalid-direction WM and all GM remain isotropic. No fMRI signal
    enters the likelihood. Observation noise is added after blur/voxel averaging.
    """
    parameter_names = PARAMETER_NAMES

    def __init__(self, pilot, design=PAT16RecoveryDesign()):
        self.pilot, self.design = pilot, design
        self.shape = pilot.mask.shape
        h = float(pilot.spacing)
        if (len(self.shape) != 3 or min(self.shape) < 2 or not pilot.mask.any()
                or not np.isfinite(h) or h <= 0
                or np.asarray(pilot.affine).shape != (4, 4)
                or not np.all(np.isfinite(pilot.affine))
                or not np.allclose(pilot.affine[:3, :3], np.eye(3)*h)
                or not np.allclose(pilot.affine[3], [0, 0, 0, 1])):
            raise ValueError("Need a nonempty three-dimensional positive RAS grid")
        if (pilot.tissue.shape != self.shape or pilot.lesion.shape != self.shape
                or pilot.anisotropic.shape != self.shape
                or pilot.directions.shape != self.shape + (3,)
                or not np.all(np.isfinite(pilot.lesion))
                or not np.all(np.isfinite(pilot.directions))
                or np.any(pilot.mask & ~np.isin(pilot.tissue, [2, 3]))
                or np.any(pilot.anisotropic & (~pilot.mask | (pilot.tissue != 3)))):
            raise ValueError("Invalid tissue, lesion or directional WM maps")
        for name in ("dt_days", "voxel_mm", "blur_sigma_mm"):
            value = getattr(design, name)
            if not np.isfinite(value) or value < 0 or (name != "blur_sigma_mm" and value == 0):
                raise ValueError(f"Invalid {name}")
        for name in ("observation_voxels", "quadrature_points"):
            value = getattr(design, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.active = np.flatnonzero(pilot.mask)
        self.initial = np.ascontiguousarray(initial_density(
            pilot, design.lesion_threshold, design.initial_amplitude).ravel()[self.active])
        if not self.initial.any():
            raise ValueError("Initialization has no lesion density in the tissue domain")
        center = design.observation_center_mm
        if center is None:
            voxels = np.argwhere((pilot.lesion > design.lesion_threshold) & pilot.mask)
            center = pilot.affine[:3, :3] @ voxels.mean(axis=0) + pilot.affine[:3, 3]
        self.observation_center_mm = np.asarray(center, float)
        if self.observation_center_mm.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Observation center must contain three finite RAS coordinates")
        m, q = design.observation_voxels, design.quadrature_points
        offsets = (np.arange(m) + .5 - m/2)*design.voxel_mm
        self.observation_axes = [c + offsets for c in self.observation_center_mm]
        # Require the entire physical voxel support to be inside the image.
        lower, upper = self.observation_center_mm - m*design.voxel_mm/2, self.observation_center_mm + m*design.voxel_mm/2
        if (np.any(lower < pilot.affine[:3, 3] - h/2)
                or np.any(upper > pilot.affine[:3, 3] + (np.asarray(self.shape)-.5)*h)):
            raise ValueError("Observation voxels must lie inside the image")
        centers = np.stack(np.meshgrid(*self.observation_axes, indexing="ij"), axis=-1)
        sub = (np.arange(q) + .5 - q/2)*design.voxel_mm/q
        quadrature = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), axis=-1).reshape(-1, 3)
        world = centers.reshape(-1, 1, 3) + quadrature[None, :, :]
        self.sample_indices = np.moveaxis((world-pilot.affine[:3, 3])/h, -1, 0)
        self.observation_shape = (m, m, m)
        self._transport_key, self._transport_value = None, None

    def transport(self, d_white, d_gray, ratio):
        if not np.all(np.isfinite([d_white, d_gray])) or min(d_white, d_gray) < 0:
            raise ValueError("Diffusion rates must be finite and nonnegative")
        key = (float(d_white), float(d_gray), float(ratio))
        if key != self._transport_key:
            diffusion = np.where(self.pilot.tissue == 3, d_white, d_gray)*self.pilot.mask
            tensor = migration_tensor(diffusion, self.pilot.directions, ratio, self.pilot.anisotropic)
            matrix, active = transport_matrix(tensor, self.pilot.mask, self.pilot.spacing)
            if not np.array_equal(active, self.active):
                raise RuntimeError("Transport and initialization domains differ")
            self._transport_key, self._transport_value = key, matrix
        return self._transport_value

    def states(self, d_white, d_gray, rho, ratio, times):
        times = np.asarray(times, float)
        if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
                or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ValueError("times must be finite, nonnegative and strictly increasing")
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("Proliferation must be finite and nonnegative")
        matrix = self.transport(d_white, d_gray, ratio)
        degree = float((-matrix.diagonal()).max())
        dt = min(self.design.dt_days, .9/degree) if degree else self.design.dt_days
        states = _states64(matrix.indptr, matrix.indices, matrix.data,
                           self.initial, times, rho, dt)
        if not np.all(np.isfinite(states)) or states.min() < -1e-12 or states.max() > 1+1e-12:
            raise RuntimeError("Density bounds violated; no clipping is performed")
        return states

    def observe(self, states):
        states = np.asarray(states, float)
        if states.ndim != 2 or states.shape[1] != len(self.active):
            raise ValueError("Expected fields with shape (time, active tissue voxels)")
        result = []
        field = np.zeros(self.shape)
        for state in states:
            field.ravel()[self.active] = state
            smooth = gaussian_filter(field, self.design.blur_sigma_mm/self.pilot.spacing,
                                     mode="constant", cval=0.)
            samples = map_coordinates(smooth, self.sample_indices, order=1,
                                      mode="nearest", prefilter=False)
            result.append(samples.mean(axis=-1).reshape(self.observation_shape))
        return np.asarray(result)

    def predict(self, d_white, d_gray, rho, ratio, times):
        return self.observe(self.states(d_white, d_gray, rho, ratio, times))


def fit_pat16_parameters(experiment, times, observations, noise_sd, *,
                         start=(.4, .15, .035, 2.), bounds=DEFAULT_BOUNDS,
                         confidence=.95):
    """Gaussian MLE in four log parameters, with known-noise local intervals.

    Local information must have full rank. Bounds are search limits, not priors;
    boundary optima are flagged and their Wald intervals are not calibrated.
    """
    if not np.isfinite(noise_sd) or noise_sd <= 0 or not 0 < confidence < 1:
        raise ValueError("Need positive noise SD and confidence in (0,1)")
    start, bounds = np.asarray(start, float), np.asarray(bounds, float)
    if (start.shape != (4,) or bounds.shape != (2, 4)
            or not np.all(np.isfinite(start)) or not np.all(np.isfinite(bounds))
            or np.any(bounds <= 0) or np.any(bounds[0] >= bounds[1])
            or np.any(start <= bounds[0]) or np.any(start >= bounds[1])
            or bounds[0, 3] < 1 or bounds[1, 3] > 3.5):
        raise ValueError("Need four positive parameters, enclosing bounds, and ratio bounds in [1,3.5]")
    times = np.asarray(times, float)
    if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite, nonnegative and strictly increasing")
    observations = np.asarray(observations, float)
    if observations.shape != (len(times), *experiment.observation_shape) or not np.all(np.isfinite(observations)):
        raise ValueError("Invalid observation shape or nonfinite observations")

    def residual(log_parameters):
        return ((experiment.predict(*np.exp(log_parameters), times)-observations)/noise_sd).ravel()

    fit = least_squares(residual, np.log(start), bounds=np.log(bounds),
                        jac="3-point", diff_step=1e-4, x_scale="jac",
                        ftol=1e-8, xtol=1e-8, gtol=1e-6)
    if not fit.success:
        raise RuntimeError(f"PAT16 fit failed: {fit.message}")
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    if len(singular) != 4 or singular[-1] <= singular[0]*1e-8:
        raise ValueError("WM diffusion, GM diffusion, proliferation and anisotropy are not locally identifiable")
    covariance = (vt.T/singular**2) @ vt
    sd = np.sqrt(np.diag(covariance))
    intervals = np.exp(fit.x[:, None] + norm.ppf(.5+confidence/2)*sd[:, None]*[-1, 1])
    at_bound = np.any(np.minimum(fit.x-np.log(bounds[0]), np.log(bounds[1])-fit.x) < 1e-4)
    return PAT16Fit(np.exp(fit.x), covariance, intervals, covariance/np.outer(sd, sd),
                    singular, fit.fun, bool(at_bound), fit.nfev)
