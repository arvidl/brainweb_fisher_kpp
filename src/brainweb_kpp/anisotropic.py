"""Monotone, conservative 19-point tensor diffusion on an isotropic RAS grid.

For diagonally dominant tensors D, decompose D into nonnegative rank-one
terms along the three axes and six face diagonals. Unlike axis-only fluxes,
this retains all mixed derivatives. Harmonic edge conductances and a CFL
bound give a convex diffusion update; exact logistic reaction preserves [0,1].
The stair-step tissue boundary has zero graph flux. Diagonal edges require
both intervening corner voxels to be tissue, preventing corner shortcuts.
"""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix


def migration_tensor(d_mean, directions, ratio=2.5, eligible=None):
    """Trace-matched prolate tensor; direction components are physical RAS.

    ratio <= 3.5 guarantees diagonal dominance for every orientation. The
    restriction belongs to this monotone stencil, not to tumor biology.
    """
    if not np.isfinite(ratio) or not 1 <= ratio <= 3.5:
        raise ValueError("anisotropy ratio must lie in [1, 3.5] for this stencil")
    d_mean = np.asarray(d_mean, float)
    v = np.asarray(directions, float)
    if v.shape != d_mean.shape + (3,) or not np.all(np.isfinite(v)):
        raise ValueError("finite RAS directions must match diffusion grid")
    if not np.all(np.isfinite(d_mean)) or np.any(d_mean < 0):
        raise ValueError("mean diffusivity must be finite and nonnegative")
    norm = np.linalg.norm(v, axis=-1)
    use = norm > 1e-8
    if eligible is not None:
        use &= np.asarray(eligible, bool)
    unit = v / np.maximum(norm[..., None], 1e-15)
    r = np.where(use, ratio, 1.)
    perpendicular = 3 * d_mean / (r + 2)
    return (perpendicular[..., None, None] * np.eye(3)
            + (perpendicular * (r - 1))[..., None, None]
            * unit[..., :, None] * unit[..., None, :])


def tensor_stencil(tensor):
    d = np.asarray(tensor, float)
    if d.shape[-2:] != (3, 3) or not np.all(np.isfinite(d)):
        raise ValueError("tensor must be finite with final dimensions 3,3")
    if not np.allclose(d, np.swapaxes(d, -1, -2), atol=1e-12):
        raise ValueError("tensor must be symmetric")
    diagonal = np.diagonal(d, axis1=-2, axis2=-1).copy()
    offsets, values = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cross = d[..., i, j]
        diagonal[..., i] -= np.abs(cross)
        diagonal[..., j] -= np.abs(cross)
        for sign in (1, -1):
            e = np.zeros(3, int); e[i] = 1; e[j] = sign
            offsets.append(e)
            values.append(np.maximum(sign * cross, 0))
    if diagonal.min() < -1e-10:
        raise ValueError("tensor is not diagonally dominant; use a wider stencil")
    for i in range(3):
        offsets.append(np.eye(3, dtype=int)[i])
        values.append(np.maximum(diagonal[..., i], 0))
    return np.asarray(offsets), np.asarray(values)


def transport_matrix(tensor, mask, spacing):
    """Return CSR generator on mask voxels, and their flat grid indices."""
    mask = np.asarray(mask, bool)
    if tensor.shape != mask.shape + (3, 3) or mask.ndim != 3 or not mask.any():
        raise ValueError("tensor and nonempty 3-D mask must match")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive")
    offsets, coefficients = tensor_stencil(tensor)
    active = np.flatnonzero(mask)
    indices = np.full(mask.shape, -1, np.int32)
    indices.flat[active] = np.arange(len(active), dtype=np.int32)
    rows, cols, vals = [], [], []
    for offset, coefficient in zip(offsets, coefficients):
        a = tuple(slice(max(0, -e), min(n, n-e)) for n, e in zip(mask.shape, offset))
        b = tuple(slice(max(0, e), min(n, n+e)) for n, e in zip(mask.shape, offset))
        valid = mask[a] & mask[b]
        # All voxels in the elementary diagonal square must be tissue.
        axes = np.flatnonzero(offset)
        if len(axes) == 2:
            for axis in axes:
                corner = list(a)
                corner[axis] = b[axis]
                valid &= mask[tuple(corner)]
        ca, cb = coefficient[a], coefficient[b]
        conductance = np.divide(2*ca*cb, ca+cb, out=np.zeros_like(ca), where=ca+cb > 0)
        valid &= conductance > 0
        i, j, w = indices[a][valid], indices[b][valid], conductance[valid] / spacing**2
        rows.extend((i, j)); cols.extend((j, i)); vals.extend((w, w))
    row, col, val = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    degree = np.bincount(row, weights=val, minlength=len(active))
    n = np.arange(len(active))
    matrix = coo_matrix((np.r_[val, -degree], (np.r_[row, n], np.r_[col, n])),
                        shape=(len(active), len(active))).tocsr()
    matrix.sort_indices()
    return matrix, active


@njit(cache=True)
def _evolve(indptr, indices, data, initial, times, rho, dt_max):
    u, work = initial.copy(), initial.copy()
    states = np.empty((len(times), len(u)), np.float32)
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


def simulate(matrix, initial, times, rho=.04, dt=.5):
    u, times = np.asarray(initial, float), np.asarray(times, float)
    if (u.shape != (matrix.shape[0],) or not np.all(np.isfinite(u))
            or u.min() < 0 or u.max() > 1):
        raise ValueError("initial normalized density must lie in [0,1]")
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("output times must be nonnegative and strictly increasing")
    if not np.isfinite(rho) or rho < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("rho must be nonnegative and dt positive")
    max_degree = float((-matrix.diagonal()).max())
    actual_dt = min(dt, .9 / max_degree) if max_degree else dt
    states = _evolve(matrix.indptr, matrix.indices, matrix.data, u, times, rho, actual_dt)
    if not np.all(np.isfinite(states)) or states.min() < -1e-7 or states.max() > 1 + 1e-7:
        raise RuntimeError("density bound violation; no clipping is performed")
    return states, actual_dt
