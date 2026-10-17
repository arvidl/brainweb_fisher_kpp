"""Export compact, auditable PAT16 results; raw MRI and the full cache stay local."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brainweb_kpp.pat16 import find_source, load_pilot, sha256


def notebook_code_hash(path):
    notebook = json.loads(path.read_text())
    source = '\n\n'.join(''.join(c['source']) for c in notebook['cells']
                         if c['cell_type'] == 'code')
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    output = ROOT / 'outputs/pat16_pilot'
    provenance = json.loads((output / 'provenance.json').read_text())
    source = find_source()
    for name, entry in provenance['source_files'].items():
        if sha256(source / entry['path']) != entry['sha256']:
            raise RuntimeError(f'Changed MRI input: {name}; rerun the notebook first')
    for name, digest in provenance['software_sha256'].items():
        if sha256(ROOT / 'src/brainweb_kpp' / name) != digest:
            raise RuntimeError(f'Changed solver/input code: {name}')
    pilot = load_pilot(source, spacing=provenance['spacing_mm'])
    cache = np.load(output / 'ensemble_cache.npz')
    states = cache['states']
    np.savez_compressed(output / 'sample_metrics.npz', **{
        key: cache[key] for key in ['samples', 'times', 'dt', 'seed', 'burden_ml',
                                    'volume_ml', 'roi_mean_density', 'threshold', 'fc', 'roi_ids']})
    center = np.rint(np.argwhere(pilot.lesion > .5).mean(axis=0)).astype(int)
    index = int(center[2])
    centers = []
    for roi in cache['roi_ids']:
        voxels = np.argwhere(pilot.atlas == roi)
        centers.append(pilot.affine[:3, :3] @ voxels.mean(axis=0) + pilot.affine[:3, 3])
    np.savez_compressed(output / 'summary_fields.npz',
                        density_quantiles=np.quantile(states, [.05, .5, .95], axis=0).astype('float32'),
                        paired_median_difference=np.median(states[:, 1] - states[:, 0], axis=0),
                        anisotropic_exceedance_fraction=(states[:, 1] >= .16).mean(axis=0).astype('float32'),
                        active=cache['active'], times=cache['times'], shape=np.array(pilot.mask.shape),
                        affine=pilot.affine, quantiles=np.array([.05, .5, .95]),
                        t1_axial=pilot.t1[:, :, index].T, lesion_axial=pilot.lesion[:, :, index].T,
                        axial_index=index, roi_centers_RAS_mm=np.array(centers),
                        lesion_center_RAS_mm=pilot.affine[:3, :3] @ center + pilot.affine[:3, 3])
    # Normalize the notebook's CSV dialect to repository LF line endings.
    # This changes serialization only; numerical values are retained exactly.
    regional = output / 'regional_exposure.csv'
    regional.write_text(regional.read_text())
    with regional.open() as stream:
        names = {int(row['roi_id']): row['roi_name'] for row in csv.DictReader(stream)}
    with (output / 'functional_connectivity.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        labels = [f'{roi}:{names[int(roi)]}' for roi in cache['roi_ids']]
        writer.writerow(['ROI', *labels])
        for label, values in zip(labels, cache['fc']):
            writer.writerow([label, *values])
    files = ['sample_metrics.npz', 'summary_fields.npz', 'functional_connectivity.csv',
             'summary.json', 'provenance.json', 'verification.json', 'regional_exposure.csv',
             'input_qa.png', 'day90_comparison.png', 'uncertainty_and_regions.png']
    manifest = {'schema_version': 1,
                'scope': 'Real MRI geometry/connectivity; assumed untreated growth, no patient rate fitting',
                'notebook_code_sha256': notebook_code_hash(ROOT / 'pat16_anisotropic_pilot.ipynb'),
                'local_cache_sha256': sha256(output / 'ensemble_cache.npz'),
                'excluded': 'Raw MRI and full ensemble_cache.npz remain local; rerun notebook from required source derivatives',
                'directional_WM_voxels': int(pilot.anisotropic.sum()),
                'files': {name: {'sha256': sha256(output / name), 'bytes': (output / name).stat().st_size}
                          for name in files}}
    (output / 'exports.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Exported compact fields, all scenario metrics and measured FC; input/code hashes verified.')


if __name__ == '__main__':
    main()
