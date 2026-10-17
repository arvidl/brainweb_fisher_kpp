import numpy as np
import nibabel as nib
import pytest
from brainweb_kpp.anisotropic import migration_tensor, tensor_stencil, transport_matrix, simulate
from brainweb_kpp.pat16 import read_mif, resample_tensor, resample_scalar, roi_definitions, find_source


def test_trace_and_direction_sign_invariance():
    v = np.random.default_rng(3).normal(size=(7, 6, 5, 3))
    d = np.full(v.shape[:-1], .5)
    tensor = migration_tensor(d, v, 3.5)
    np.testing.assert_allclose(np.trace(tensor, axis1=-2, axis2=-1), 3*d)
    np.testing.assert_allclose(tensor, migration_tensor(d, -v, 3.5))
    eig = np.linalg.eigvalsh(tensor)
    np.testing.assert_allclose(eig[..., -1]/eig[..., 0], 3.5)
    tensor_stencil(tensor)  # Worst allowed orientation must remain monotone.


def test_full_tensor_stencil_recovers_mixed_derivatives():
    v = np.array([1., 2., -1.]); v /= np.linalg.norm(v)
    tensor = migration_tensor(np.array(.7), v, 3.)
    offsets, coefficients = tensor_stencil(tensor)
    reconstructed = np.einsum('n,ni,nj->ij', coefficients, offsets, offsets)
    np.testing.assert_allclose(reconstructed, tensor, atol=1e-14)
    # Analytic L(x_i*x_j)=2*D_ij; axis-only stencils would give zero.
    assert abs(tensor[0, 1]) > .1
    mask = np.ones((9, 9, 9), bool)
    full = np.broadcast_to(tensor, mask.shape+(3, 3))
    matrix, active = transport_matrix(full, mask, 1.)
    x, y, _ = np.indices(mask.shape)
    actual = (matrix @ (x*y).ravel()).reshape(mask.shape)
    np.testing.assert_allclose(actual[2:-2, 2:-2, 2:-2], 2*tensor[0, 1], atol=2e-14)


def test_conservation_no_flux_bounds_and_isotropic_limit():
    mask = np.ones((11, 12, 13), bool); mask[5, 5, :] = False
    v = np.broadcast_to([1., 1., .2], mask.shape+(3,))
    tensor = migration_tensor(np.full(mask.shape, .5), v, 3.)
    matrix, active = transport_matrix(tensor, mask, 2.)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)), 0, atol=1e-15)
    np.testing.assert_allclose((matrix-matrix.T).data, 0, atol=1e-15)
    initial = np.random.default_rng(6).uniform(0, .8, len(active))
    states, dt = simulate(matrix, initial, [0, 1.3, 4.8], rho=0, dt=100)
    np.testing.assert_allclose(states.sum(axis=1), initial.sum(), rtol=2e-7)
    assert dt * (-matrix.diagonal()).max() <= .9 + 1e-12
    assert states.min() >= 0 and states.max() <= .8
    iso = migration_tensor(np.full(mask.shape, .5), v, 1)
    identity = .5 * np.broadcast_to(np.eye(3), tensor.shape)
    np.testing.assert_allclose(iso, identity)
    m2, _ = transport_matrix(iso, mask, 2.)
    # Each axis edge has D/h², and no diagonal edges exist in the reference.
    offdiag = m2 - __import__('scipy').sparse.diags(m2.diagonal())
    np.testing.assert_allclose(offdiag.data, .125)


def test_no_diagonal_corner_shortcuts():
    mask = np.zeros((3, 3, 3), bool)
    mask[0, 0, 1] = mask[1, 1, 1] = True
    tensor = migration_tensor(np.ones(mask.shape), np.broadcast_to([1., 1., 0.], mask.shape+(3,)), 3)
    matrix, _ = transport_matrix(tensor, mask, 1.)
    assert matrix.nnz == 2 and np.all(matrix.data == 0)


def test_heterogeneous_isotropic_operator_matches_existing_solver():
    from brainweb_kpp.model import face_coefficients
    rng = np.random.default_rng(16)
    shape = (6, 7, 8)
    mask = rng.uniform(size=shape) > .1
    diffusion = rng.uniform(.05, .6, shape) * mask
    tensor = diffusion[..., None, None] * np.eye(3)
    matrix, active = transport_matrix(tensor, mask, 2.)
    initial = rng.uniform(size=shape) * mask
    faces, _ = face_coefficients(diffusion, mask, 2.)
    flux = np.zeros(shape).ravel()
    field = initial.ravel()
    strides = (shape[1]*shape[2], shape[2], 1)
    for axis, step in enumerate(strides):
        for i in np.flatnonzero(faces[axis]):
            f = faces[axis, i] * (field[i+step]-field[i])
            flux[i] += f; flux[i+step] -= f
    np.testing.assert_allclose(matrix @ field[active], flux[active], atol=3e-16)


def test_gaussian_spread_recovers_full_rotated_tensor():
    shape = (25, 25, 25)
    coordinates = np.indices(shape).reshape(3, -1).T - 12
    initial = .1*np.exp(-np.sum(coordinates**2, axis=1)/(2*1.2**2))
    tensor = migration_tensor(np.array(.2), np.array([1., 2., -1.]), 3.)
    matrix, _ = transport_matrix(np.broadcast_to(tensor, shape+(3,3)), np.ones(shape, bool), 1.)
    states, _ = simulate(matrix, initial, [0.,2.], rho=0, dt=.1)
    covariance = []
    for state in states:
        covariance.append(np.einsum('n,ni,nj->ij', state/state.sum(), coordinates, coordinates))
    np.testing.assert_allclose(covariance[1]-covariance[0], 4*tensor, atol=2e-6)


def test_monotone_stencil_rejects_unsupported_tensors():
    with pytest.raises(ValueError, match='ratio'):
        migration_tensor(np.array(1.), np.array([1., 0., 0.]), 8)
    with pytest.raises(ValueError, match='diagonally dominant'):
        tensor_stencil(np.array([[.1, .4, 0], [.4, 3., 0], [0, 0, 1.]]))


@pytest.mark.parametrize('layout', [('-0', '-1', '+2', '+3'), ('+2', '-0', '+3', '-1')])
def test_mif_signed_and_permuted_layout(tmp_path, layout):
    data = np.arange(2*3*4*6, dtype=np.float32).reshape(2, 3, 4, 6)
    physical = data.copy()
    for axis, value in enumerate(layout):
        if value.startswith('-'):
            physical = np.flip(physical, axis)
    axes = np.argsort([abs(int(s)) for s in layout])
    packed = physical.transpose(axes).ravel(order='F')
    header = ('mrtrix image\ndim: 2,3,4,6\nvox: 2,3,4,1\nlayout: '+','.join(layout)
              +'\ndatatype: Float32LE\ntransform: 1,0,0,5\ntransform: 0,1,0,6'
              +'\ntransform: 0,0,1,7\nfile: . 1024\nEND\n').encode()
    path = tmp_path/'tensor.mif'
    path.write_bytes(header + b'\0'*(1024-len(header)) + packed.astype('<f4').tobytes())
    actual, affine, _ = read_mif(path)
    np.testing.assert_array_equal(actual, data)
    np.testing.assert_allclose(affine, [[2,0,0,5],[0,3,0,6],[0,0,4,7],[0,0,0,1]])


def test_tensor_pullback_rotates_components_and_spatial_positions():
    # Target points pull back through a 90-degree rotation into the source.
    pullback = np.array([[0., -1, 0, 4], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    six = np.zeros((5, 5, 5, 6)); six[..., :3] = [3, 1, 1]
    actual = resample_tensor(six, np.eye(4), (5, 5, 5), np.eye(4), pullback)
    np.testing.assert_allclose(actual[..., 0, 0], 1.)
    np.testing.assert_allclose(actual[..., 1, 1], 3.)
    # A scalar marker also obeys the world-coordinate pullback.
    six[4, 1, 2, 0] = 7
    actual = resample_tensor(six, np.eye(4), (5, 5, 5), np.eye(4), pullback)
    assert actual[1, 0, 2, 1, 1] == 7


def test_resampling_checks_affine_even_when_shapes_match():
    data = np.zeros((5, 5, 5)); data[2, 2, 2] = 1
    affine = np.eye(4); affine[0, 3] = 1
    result = resample_scalar(nib.Nifti1Image(data, affine), data.shape, np.eye(4), 0)
    assert result[3, 2, 2] == 1 and result[2, 2, 2] == 0


def test_dk_roi_identity():
    names = roi_definitions()
    assert len(names) == 82
    assert names[1024] == 'L-precentral' and names[2035] == 'R-insula'
    assert names[53] == 'R-Hippocampus' and 1004 not in names


def test_explicit_data_path_does_not_silently_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match='PAT16_SOURCE'):
        find_source(tmp_path/'missing-project')
