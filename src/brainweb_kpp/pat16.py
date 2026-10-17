"""PAT16 multimodal pilot inputs. No synthetic imaging fallback or fitted rates.

Only pre-operative data are used. All spatial arrays are resampled using their
affines into one axis-aligned physical RAS grid. MRtrix tensors use scanner-RAS
components; the saved MRtrix registration maps T1 target points to ACPC source
points (pullback), so its inverse rotation reorients tensor components.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import affine_transform, gaussian_filter


def read_mif(path):
    """Read embedded, uncompressed Float32LE MIF with arbitrary signed layout.

    Intentionally fail on other formats rather than guess their interpretation.
    Negative zero in MIF layouts is meaningful and must remain a string.
    """
    header = {}
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'mrtrix image':
            raise ValueError('not an MRtrix image')
        for _ in range(10000):
            line = stream.readline()
            if line.strip() == b'END':
                break
            if not line:
                raise ValueError('unterminated MIF header')
            key, value = line.decode().strip().split(':', 1)
            header.setdefault(key, []).append(value.strip())
        else:
            raise ValueError('MIF header too long')
    if header['datatype'] != ['Float32LE']:
        raise ValueError('only Float32LE MIF is supported; export other formats first')
    filename, offset = header['file'][0].split()
    if filename != '.':
        raise ValueError('only embedded MIF data are supported')
    shape = tuple(map(int, header['dim'][0].split(',')))
    layout = header['layout'][0].split(',')
    ranks = list(map(lambda x: abs(int(x)), layout))
    if sorted(ranks) != list(range(len(shape))):
        raise ValueError('invalid MIF layout')
    axes = np.argsort(ranks)
    raw = np.memmap(path, dtype='<f4', mode='r', offset=int(offset), shape=(int(np.prod(shape)),))
    data = raw.reshape(tuple(shape[a] for a in axes), order='F').transpose(np.argsort(axes))
    for axis, stride in enumerate(layout):
        if stride.startswith('-'):
            data = np.flip(data, axis)
    affine = np.eye(4)
    affine[:3] = np.array([list(map(float, s.split(','))) for s in header['transform']])
    affine[:3, :3] *= np.array(list(map(float, header['vox'][0].split(',')))[:3])[None, :]
    return data, affine, header


def resample_scalar(image, shape, affine, order=1):
    return np.asarray(resample_from_to(image, (shape, affine), order=order).dataobj)


def resample_tensor(data, source_affine, target_shape, target_affine, pullback):
    """Resample six scanner-RAS components then apply forward rigid rotation."""
    pullback = np.asarray(pullback, float)
    rotation = np.linalg.inv(pullback[:3, :3])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4) or np.linalg.det(rotation) < 0:
        raise ValueError('tensor reorientation requires a proper rigid transform')
    mapping = np.linalg.inv(source_affine) @ pullback @ target_affine
    six = np.stack([affine_transform(data[..., k], mapping[:3, :3], mapping[:3, 3],
                                    output_shape=target_shape, order=1, prefilter=False)
                    for k in range(6)], axis=-1)
    tensor = np.zeros(target_shape + (3, 3), float)
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = np.moveaxis(six[..., :3], -1, 0)
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
        tensor[..., i, j] = tensor[..., j, i] = six[..., k]
    return np.einsum('ia,...ab,jb->...ij', rotation, tensor, rotation)


def source_paths(root):
    root = Path(root)
    derivatives = root / 'derivatives'
    anat = derivatives / 'fmriprep/sub-PAT16/ses-preop/anat'
    func = derivatives / 'fmriprep/sub-PAT16/ses-preop/func'
    prefix = 'sub-PAT16_ses-preop'
    return {
        't1': anat / f'{prefix}_desc-preproc_T1w.nii.gz',
        'brain': anat / f'{prefix}_desc-brain_mask.nii.gz',
        'tissue': anat / f'{prefix}_dseg.nii.gz',
        'CSF': anat / f'{prefix}_label-CSF_probseg.nii.gz',
        'GM': anat / f'{prefix}_label-GM_probseg.nii.gz',
        'WM': anat / f'{prefix}_label-WM_probseg.nii.gz',
        'lesion': root / 'data/BIDS_preop_postop/derivatives/tumor_masks/sub-PAT16/anat/sub-PAT16_space_T1_label-tumor.nii.gz',
        'tensor': derivatives / 'mrtrix_recon/sub-PAT16/ses-preop/dti.mif',
        'pullback': derivatives / 'registered_t1w/sub-PAT16/ses-preop/transforms/acpc_to_t1w_mrtrix.txt',
        'fa_registered': derivatives / 'registered_t1w/sub-PAT16/ses-preop/fa_space-T1w.nii.gz',
        'parcellation': derivatives / 'registered_t1w/sub-PAT16/ses-preop/aparc+aseg_space-T1w.nii.gz',
        'bold': func / f'{prefix}_task-rest_space-T1w_desc-preproc_bold.nii.gz',
        'bold_json': func / f'{prefix}_task-rest_space-T1w_desc-preproc_bold.json',
        'confounds': func / f'{prefix}_task-rest_desc-confounds_timeseries.tsv',
    }


def find_source(explicit=None):
    configured = explicit or os.environ.get('PAT16_SOURCE', '')
    candidates = ([configured] if configured else
                  [Path.home() / 'prj/Finse_2026/glioma-brain-digital-twin_20261011_0800'])
    for candidate in candidates:
        if candidate and all(p.is_file() for p in source_paths(candidate).values()):
            return Path(candidate).resolve()
    raise FileNotFoundError('Set PAT16_SOURCE to the processed digital-twin project root. '
                            'See examples/pat16_pilot.md for the required real inputs; no synthetic fallback is used.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PilotData:
    t1: np.ndarray
    tissue: np.ndarray
    lesion: np.ndarray
    mask: np.ndarray
    directions: np.ndarray
    fa: np.ndarray
    anisotropic: np.ndarray
    atlas: np.ndarray
    affine: np.ndarray
    spacing: float
    provenance: dict


def load_pilot(root, spacing=3.):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive')
    paths = source_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    brain = nib.load(paths['brain'])
    points = nib.affines.apply_affine(brain.affine, np.argwhere(brain.get_fdata() > .5))
    lower = np.floor((points.min(axis=0) - spacing) / spacing) * spacing
    upper = np.ceil((points.max(axis=0) + spacing) / spacing) * spacing
    shape = tuple(np.round((upper-lower)/spacing).astype(int) + 1)
    affine = np.diag([spacing, spacing, spacing, 1.]); affine[:3, 3] = lower
    sample = lambda key, order=1: resample_scalar(nib.load(paths[key]), shape, affine, order)
    t1 = sample('t1').astype(np.float32)
    brainmask = sample('brain', 0) > .5
    t1 /= np.percentile(t1[brainmask], 99.5)
    # Use named probability maps: integer dseg conventions vary by pipeline.
    # This particular derivative labels GM=1, WM=2, CSF=3, unlike BrainWeb.
    probability = np.stack([sample(key) for key in ('CSF', 'GM', 'WM')])
    tissue = (np.argmax(probability, axis=0) + 1).astype(np.uint8)
    tissue[~brainmask] = 0
    lesion = np.clip(sample('lesion'), 0, 1).astype(np.float32)
    # Label-positive lesion voxels may be mislabeled as CSF by a healthy-tissue
    # segmenter. Include them explicitly, with assumed gray-matter transport.
    overrides = brainmask & (lesion > .2) & ~np.isin(tissue, [2, 3])
    tissue[overrides] = 2
    mask = brainmask & np.isin(tissue, [2, 3])
    if not (lesion[mask] > .5).any():
        raise ValueError('lesion does not overlap the tissue domain; check registration')
    raw, source_affine, header = read_mif(paths['tensor'])
    pullback = np.loadtxt(paths['pullback'])
    tensor = resample_tensor(raw, source_affine, shape, affine, pullback)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    fa = np.sqrt(1.5 * np.sum((eigenvalues - eigenvalues.mean(axis=-1)[..., None])**2, axis=-1)
                 / np.maximum(np.sum(eigenvalues**2, axis=-1), 1e-30))
    valid = (eigenvalues[..., 0] > 0) & (fa > .2) & (fa <= 1.) & mask
    directions = eigenvectors[..., :, -1].astype(np.float32)
    directions[~valid] = 0
    registered = sample('fa_registered')
    check = mask & (registered > .1) & (eigenvalues[..., 0] > 0)
    correlation = float(np.corrcoef(fa[check], registered[check])[0, 1])
    if not np.isfinite(correlation) or correlation < .85:
        raise ValueError(f'FA alignment check failed (r={correlation:.3f})')
    fa = np.where(valid, fa, 0).astype(np.float32)
    atlas = sample('parcellation', 0).astype(np.int16)
    provenance = {
        'subject': 'sub-PAT16', 'session': 'ses-preop', 'dataset': 'OpenNeuro ds001226 (BTC)',
        'source_repository': 'https://github.com/arvidl/glioma-brain-digital-twin-public',
        'source_files': {key: {'path': str(path.relative_to(root)), 'sha256': sha256(path)}
                         for key, path in paths.items()},
        'grid_shape': list(map(int, shape)), 'affine_RAS_mm': affine.tolist(), 'spacing_mm': spacing,
        'active_voxels': int(mask.sum()), 'lesion_tissue_override_voxels': int(overrides.sum()),
        'internal_tissue_labels': {'CSF': 1, 'GM': 2, 'WM': 3},
        'tissue_method': 'argmax of named CSF/GM/WM probability maps; no assumed dseg label ordering',
        'fa_alignment_correlation': correlation,
        'tensor_basis': 'MRtrix scanner RAS; xx yy zz xy xz yz',
        'registration': 'MRtrix T1-to-ACPC pullback; inverse rigid rotation applied to tensor',
        'dti_history': header.get('command_history', []),
        'dti_b_shells_s_per_mm2': sorted(set(round(float(s.split(',')[-1]))
                                            for s in header.get('prior_dw_scheme', []))),
        'interpretation': 'Real anatomy/dMRI/fMRI; conditional synthetic growth, no patient rate fit',
    }
    return PilotData(t1, tissue, lesion, mask, directions, fa, valid & (tissue == 3),
                     atlas, affine, spacing, provenance)


CORTICAL_NAMES = ('bankssts caudalanteriorcingulate caudalmiddlefrontal unknown cuneus entorhinal '
                  'fusiform inferiorparietal inferiortemporal isthmuscingulate lateraloccipital '
                  'lateralorbitofrontal lingual medialorbitofrontal middletemporal parahippocampal '
                  'paracentral parsopercularis parsorbitalis parstriangularis pericalcarine '
                  'postcentral posteriorcingulate precentral precuneus rostralanteriorcingulate '
                  'rostralmiddlefrontal superiorfrontal superiorparietal superiortemporal '
                  'supramarginal frontalpole temporalpole transversetemporal insula').split()


def roi_definitions():
    sub = {10:'Thalamus', 11:'Caudate', 12:'Putamen', 13:'Pallidum',
           17:'Hippocampus', 18:'Amygdala', 26:'Accumbens'}
    names = {k: 'L-'+name for k, name in sub.items()}
    names.update({k+39: 'R-'+name for k, name in sub.items() if k not in (17, 18, 26)})
    names.update({53:'R-Hippocampus', 54:'R-Amygdala', 58:'R-Accumbens'})
    for side, base in (('L', 1000), ('R', 2000)):
        names.update({base+i: side+'-'+name for i, name in enumerate(CORTICAL_NAMES, 1)
                      if name != 'unknown'})
    return names


def functional_network(root, min_voxels=8, fd_threshold=.5):
    """Recompute ROI correlations from real preprocessed BOLD, never stale FC.

    Joint OLS nuisance model on retained frames: motion plus derivatives and
    squares (24), first five aCompCor components, all supplied high-pass cosines,
    linear trend and intercept. FD>0.5mm and nonsteady frames are censored.
    No additional temporal filter or global-signal regression is applied.
    """
    paths = source_paths(root)
    bold = nib.load(paths['bold'])
    atlas = resample_scalar(nib.load(paths['parcellation']), bold.shape[:3], bold.affine, 0)
    data = bold.get_fdata(dtype=np.float32)
    names = roi_definitions()
    ids = np.array([i for i in sorted(names) if (atlas == i).sum() >= min_voxels])
    if len(ids) < 20:
        raise ValueError('insufficient atlas/BOLD overlap')
    timeseries = np.array([data[atlas == i].mean(axis=0) for i in ids]).T
    with paths['confounds'].open() as stream:
        table = list(csv.DictReader(stream, delimiter='\t'))
    if len(table) != len(timeseries):
        raise ValueError('BOLD and confound frame counts differ')
    keys = list(table[0])
    motion = ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')
    columns = [k for k in keys if any(k == m or k.startswith(m+'_') for m in motion)]
    columns += sorted(k for k in keys if k.startswith('a_comp_cor_'))[:5]
    columns += [k for k in keys if k.startswith('cosine')]
    if not all(m in columns for m in motion) or 'framewise_displacement' not in keys:
        raise ValueError('required motion confounds are missing')
    def numeric(key):
        return np.array([float(r[key]) if r[key] not in ('n/a', '') else np.nan for r in table])
    retained = np.nan_to_num(numeric('framewise_displacement')) <= fd_threshold
    for key in keys:
        if key.startswith('non_steady_state_outlier'):
            retained &= numeric(key) == 0
    if retained.sum() < 60:
        raise ValueError('fewer than 60 retained BOLD frames; inspect motion QC')
    nuisance = np.array([np.nan_to_num(numeric(k)) for k in columns]).T[retained]
    scale = nuisance.std(axis=0)
    nuisance = (nuisance[:, scale > 1e-12] - nuisance[:, scale > 1e-12].mean(axis=0)) / scale[scale > 1e-12]
    design = np.column_stack([np.ones(retained.sum()), np.linspace(-1, 1, len(retained))[retained], nuisance])
    rank = int(np.linalg.matrix_rank(design))
    if retained.sum() - rank < 30:
        raise ValueError('insufficient residual degrees of freedom for FC')
    y = timeseries[retained]
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    valid = residual.std(axis=0) > 1e-6
    ids, residual = ids[valid], residual[:, valid]
    fc = np.corrcoef(residual.T)
    np.fill_diagonal(fc, 0)
    with paths['bold_json'].open() as stream:
        tr = json.load(stream)['RepetitionTime']
    return {'ids': ids, 'names': [names[i] for i in ids], 'fc': fc, 'residual': residual,
            'qc': {'total_frames': len(retained), 'retained_frames': int(retained.sum()),
                   'confound_rank': rank, 'residual_dof': int(retained.sum())-rank,
                   'fd_threshold_mm': fd_threshold, 'TR_seconds': tr,
                   'regressors': columns, 'global_signal_regression': False}}


def initial_density(pilot, threshold=.5, amplitude=.6):
    if not 0 < threshold < 1 or not 0 < amplitude <= 1:
        raise ValueError('lesion threshold in (0,1) and density amplitude in (0,1] required')
    u = gaussian_filter((pilot.lesion > threshold).astype(float), .7)
    return amplitude * u * pilot.mask
