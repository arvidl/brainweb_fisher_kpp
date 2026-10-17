"""Small, homogeneous 3-D benchmark for Fisher–KPP parameter recovery.

Uses the explorer's conservative diffusion / exact logistic reaction kernel,
in float64 for finite-difference optimization. This is a calibrated synthetic
imaging experiment, not a conversion from clinical MRI intensity to cell count.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.stats import norm

from .model import advance, face_coefficients


@dataclass(frozen=True)
class Phantom:
    width_mm: float = 60.0
    dx_mm: float = 2.0
    dt_days: float = 0.25
    voxel_mm: float = 6.0
    blur_sigma_mm: float = 2.0
    seed_sigma_mm: tuple = (5.0, 4.0, 5.0)
    seed_center_mm: tuple = (-3.0, 2.0, 0.0)
    seed_peak: float = 0.2


class SyntheticExperiment:
    """Homogeneous, no-flux box; u = viable density / known capacity K.

    A Gaussian initial condition is integrated over each finite-volume cell.
    Observations blur the field and average disjoint, aligned MRI-like voxels.
    Independent Gaussian measurement noise must be added *after* this operator.
    """

    def __init__(self, phantom=Phantom()):
        p = phantom
        for name in ("width_mm", "dx_mm", "dt_days", "voxel_mm", "blur_sigma_mm", "seed_peak"):
            value = getattr(p, name)
            if not np.isfinite(value) or value < 0 or (name != "blur_sigma_mm" and value == 0):
                raise ValueError(f"Invalid {name}")
        if p.seed_peak > 1:
            raise ValueError("seed_peak must be at most one")
        if len(p.seed_sigma_mm) != 3 or len(p.seed_center_mm) != 3:
            raise ValueError("Seed widths and center must have three coordinates")
        if not np.all(np.isfinite(p.seed_center_mm)) or not np.all(np.isfinite(p.seed_sigma_mm)) or min(p.seed_sigma_mm) <= 0:
            raise ValueError("Seed coordinates must be finite and widths positive")
        n, block = round(p.width_mm / p.dx_mm), round(p.voxel_mm / p.dx_mm)
        if n < 2 or block < 1 or n % block or not np.isclose(n * p.dx_mm, p.width_mm) or not np.isclose(block * p.dx_mm, p.voxel_mm):
            raise ValueError("Domain and observation voxels must align with the simulation grid")
        self.phantom, self.shape, self.block = p, (n, n, n), block
        self.coordinates = (np.arange(n) + 0.5) * p.dx_mm - p.width_mm / 2
        initial = np.full(self.shape, p.seed_peak, dtype=float)
        for axis, (sigma, center) in enumerate(zip(p.seed_sigma_mm, p.seed_center_mm)):
            x = self.coordinates - center
            average = sigma * np.sqrt(np.pi / 2) / p.dx_mm * (
                erf((x + p.dx_mm / 2) / (np.sqrt(2) * sigma))
                - erf((x - p.dx_mm / 2) / (np.sqrt(2) * sigma)))
            shape = [1, 1, 1]
            shape[axis] = n
            initial *= average.reshape(shape)
        self.initial = np.ascontiguousarray(initial)
        self.unit_faces, self.unit_safe_dt = face_coefficients(
            np.ones(self.shape), np.ones(self.shape, dtype=bool), p.dx_mm)
        self.active = np.arange(initial.size)
        self.strides = np.array([n * n, n, 1])
        self.capacity = np.ones(initial.size)

    def states(self, diffusion, proliferation, times):
        """Return normalized 3-D fields at strictly increasing nonnegative days."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, nonnegative and strictly increasing")
        if not np.isfinite(diffusion) or not np.isfinite(proliferation) or min(diffusion, proliferation) < 0:
            raise ValueError("Rates must be finite and nonnegative")
        if diffusion * self.phantom.dt_days > self.unit_safe_dt * (1 + 1e-12):
            raise ValueError("Requested time step violates the diffusion stability limit")
        faces = self.unit_faces * diffusion
        n, out = self.initial.ravel().copy(), np.empty(self.initial.size)
        snapshots = np.empty((times.size, *self.shape))
        previous = 0.0
        for index, target in enumerate(times):
            interval = target - previous
            steps = max(1, math.ceil(interval / self.phantom.dt_days))
            dt = interval / steps
            decay = np.full(n.size, np.exp(-proliferation * dt))
            for _ in range(steps):
                advance(n, out, faces, self.active, self.strides, decay, self.capacity, dt)
                n, out = out, n
            snapshots[index] = n.reshape(self.shape)
            previous = target
        return snapshots

    def observe(self, states):
        """Gaussian point-spread function, then physical voxel averaging."""
        states = np.asarray(states, dtype=float)
        if states.ndim != 4 or states.shape[1:] != self.shape:
            raise ValueError("Expected fields with shape (time, x, y, z)")
        b, m = self.block, self.shape[0] // self.block
        result = []
        for volume in states:
            smooth = gaussian_filter(volume, self.phantom.blur_sigma_mm / self.phantom.dx_mm,
                                     mode="reflect")
            result.append(smooth.reshape(m, b, m, b, m, b).mean(axis=(1, 3, 5)))
        return np.asarray(result)

    def predict(self, diffusion, proliferation, times):
        return self.observe(self.states(diffusion, proliferation, times))


@dataclass
class RecoveryFit:
    parameters: np.ndarray             # D [mm²/day], rho [1/day]
    log_covariance: np.ndarray         # known-noise local information approximation
    intervals: np.ndarray              # rows = parameters; columns = lower, upper
    correlation: float
    standardized_residuals: np.ndarray
    singular_values: np.ndarray
    at_bound: bool
    evaluations: int


def fit_parameters(experiment, times, observations, noise_sd, *, start=(0.2, 0.03),
                   bounds=((0.05, 0.01), (0.8, 0.09)), confidence=0.95):
    """Bounded Gaussian maximum likelihood in log(D), log(rho).

    noise_sd is the *known*, independent, post-imaging noise SD. The local
    intervals are asymptotic confidence intervals, not Bayesian posteriors.
    Rank-deficient information is rejected rather than silently inverted.
    A boundary optimum is returned with at_bound=True: its Wald intervals
    must not be interpreted as calibrated confidence intervals.
    """
    if not np.isfinite(noise_sd) or noise_sd <= 0 or not 0 < confidence < 1:
        raise ValueError("Need positive noise SD and confidence between zero and one")
    start, bounds = np.asarray(start, dtype=float), np.asarray(bounds, dtype=float)
    if start.shape != (2,) or bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds <= 0) or np.any(bounds[0] >= bounds[1]) or not np.all(np.isfinite(start)) or np.any(start <= bounds[0]) or np.any(start >= bounds[1]):
        raise ValueError("Provide two positive rates and strictly enclosing positive bounds")
    observations = np.asarray(observations, dtype=float)
    expected_shape = (len(times),) + (experiment.shape[0] // experiment.block,) * 3
    if observations.shape != expected_shape or not np.all(np.isfinite(observations)):
        raise ValueError(f"observations must be finite with shape {expected_shape}")

    def residual(log_parameters):
        predicted = experiment.predict(*np.exp(log_parameters), times)
        return ((predicted - observations) / noise_sd).ravel()

    result = least_squares(residual, np.log(start), bounds=np.log(bounds),
                           jac="3-point", diff_step=1e-4, x_scale="jac",
                           ftol=1e-8, xtol=1e-8, gtol=1e-6)
    if not result.success:
        raise RuntimeError(f"Parameter fit failed: {result.message}")
    _, singular, vt = np.linalg.svd(result.jac, full_matrices=False)
    if len(singular) < 2 or singular[-1] <= singular[0] * 1e-8:
        raise ValueError("D and rho are not locally identifiable in this experiment")
    covariance = (vt.T / singular**2) @ vt
    sd = np.sqrt(np.diag(covariance))
    z = norm.ppf(0.5 + confidence / 2)
    intervals = np.exp(result.x[:, None] + z * sd[:, None] * np.array([-1, 1]))
    near_bound = np.any(np.minimum(result.x - np.log(bounds[0]),
                                   np.log(bounds[1]) - result.x) < 1e-4)
    return RecoveryFit(np.exp(result.x), covariance, intervals,
                       covariance[0, 1] / np.prod(sd), result.fun, singular,
                       bool(near_bound), result.nfev)
