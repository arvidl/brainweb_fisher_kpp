"""Run the PAT16 synthetic recovery audit without a Jupyter kernel.

From brainweb_fisher_kpp: uv run --locked python scripts/run_pat16_recovery.py
Real PAT16 derivatives are required; no synthetic anatomy fallback is used.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import os
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/brainweb-mpl')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from brainweb_kpp.pat16 import find_source, load_pilot
    from brainweb_kpp.pat16_recovery import PAT16RecoverySettings, run_pat16_recovery

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Processed PAT16 project root (or PAT16_SOURCE)')
    parser.add_argument('--output', type=Path, default=root/'outputs/pat16_recovery')
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--coverage', type=int, default=60)
    parser.add_argument('--seed', type=int, default=161026)
    parser.add_argument('--spacing', type=float, default=3.)
    args = parser.parse_args()
    pilot = load_pilot(find_source(args.source), spacing=args.spacing)
    settings = replace(PAT16RecoverySettings(), n_bootstrap=args.bootstrap,
                       n_coverage=args.coverage, seed=args.seed)
    summary, _ = run_pat16_recovery(pilot, args.output, settings=settings,
                                   progress=lambda message: print(message, flush=True))
    print('\nParameter              Known       Estimate       Bootstrap interval', flush=True)
    for row in summary['parameters']:
        print(f"{row['parameter']:23s} {row['known']:.6g}    {row['estimate']:.6g}    "
              f"{row['bootstrap_interval']}  coverage {row['local_coverage_hits']}/{row['local_coverage_trials']}", flush=True)


if __name__ == '__main__':
    main()
