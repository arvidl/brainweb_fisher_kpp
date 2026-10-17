"""Independent nested prediction-interval audit on real PAT16 anatomy."""
from pathlib import Path
import argparse
import os
import sys


def verify_anatomy(pilot, recovery, output):
    """Require the baseline anatomy and reproduce its finer-time known signal."""
    from dataclasses import replace
    from importlib.metadata import version
    import json
    import numpy as np
    from brainweb_kpp.pat16 import sha256
    from brainweb_kpp.pat16_inference import PAT16Experiment, PAT16RecoveryDesign
    provenance = json.loads((recovery/'provenance.json').read_text())
    anatomy = dict(pilot.provenance)
    anatomy['dti_history'] = [line.split()[0] for line in anatomy.get('dti_history', [])]
    if anatomy != provenance['anatomy']:
        raise ValueError('PAT16 inputs/geometry differ from the baseline recovery anatomy')
    baseline = json.loads((recovery/'summary.json').read_text())
    design = PAT16RecoveryDesign(**baseline['design'])
    generator = PAT16Experiment(pilot, replace(design, dt_days=baseline['generator_dt_days']))
    with np.load(recovery/'recovery_data.npz') as z:
        regenerated = generator.predict(*z['known_parameters'], z['scan_days'])
        np.testing.assert_array_equal(regenerated, z['clean'])
    record = dict(anatomy=anatomy, known_signal_max_absolute_difference=0.,
                  baseline_provenance_sha256=sha256(recovery/'provenance.json'),
                  runner_sha256=sha256(Path(__file__)),
                  software_versions={name: version(name) for name in ['numpy', 'scipy', 'numba', 'nibabel']})
    output.mkdir(parents=True, exist_ok=True)
    (output/'anatomy_validation.json').write_text(json.dumps(record, indent=2)+'\n')
    return sha256(output/'anatomy_validation.json')


def complete_manifest(output, anatomy_hash):
    import json
    summary_path = output/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary['anatomy_validation_sha256'] = anatomy_hash
        summary_path.write_text(json.dumps(summary, indent=2)+'\n')


def main():
    # Keep each worker single threaded; parallelism is across complete datasets.
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
        os.environ[name] = '1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from brainweb_kpp.pat16 import find_source, load_pilot
    from brainweb_kpp.pat16_prediction import PredictionSettings, run_prediction_audit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--recovery', type=Path, default=root/'outputs/pat16_recovery')
    parser.add_argument('--output', type=Path, default=root/'outputs/pat16_prediction')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=161027)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fresh', action='store_true',
                        help='Archive an existing output directory before starting a fresh audit')
    parser.add_argument('--verify-anatomy-only', action='store_true',
                        help='Check inputs and the known signal; attach provenance to an existing audit')
    args = parser.parse_args()
    if args.fresh and args.output.exists():
        from datetime import datetime, timezone
        import shutil
        archive = args.output.with_name(args.output.name+'_archive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        shutil.move(str(args.output), str(archive))
        print(f'Archived previous records to {archive}', flush=True)
    pilot = load_pilot(find_source(args.source), spacing=3.)
    anatomy_hash = verify_anatomy(pilot, args.recovery, args.output)
    if args.verify_anatomy_only:
        complete_manifest(args.output, anatomy_hash)
        print('PAT16 anatomy/input hashes and every finer-time known signal value match the baseline.', flush=True)
        return
    run_prediction_audit(pilot, args.recovery, args.output,
                         PredictionSettings(args.trials, args.bootstrap, args.draws, args.seed),
                         workers=args.workers, progress=lambda message: print(message, flush=True))
    complete_manifest(args.output, anatomy_hash)


if __name__ == '__main__':
    main()
