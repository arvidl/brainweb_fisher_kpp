"""Reproduce the manuscript's deterministic BrainWeb experiments.

Run from any directory with the repository's installed Python environment.
No patient data, fitted parameters, or stochastic sampling are used.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np

from brainweb_kpp.data import DATA, load_brainweb, verify_data
from brainweb_kpp.model import Parameters, get_planes, simulate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    verify_data()
    config = json.loads((ROOT / "examples/coupled.json").read_text())
    p = Parameters(**config)
    brain = load_brainweb(2)
    results = {}
    records = []

    def record(name, result, spacing, settings):
        results[name] = result
        row = dict(name=name, spacing_mm=spacing, dt_days=settings.dt_days,
                   parameters=result.parameters, metadata=result.metadata,
                   final={key: float(value[-1]) for key, value in result.metrics.items()},
                   initial={key: float(value[0]) for key, value in result.metrics.items()})
        if result.final_oxygen is not None:
            row["final"]["max_necrosis_over_K"] = float(result.final_necrosis.max() / p.carrying_capacity)
            row["final"]["necrotic_voxels_20"] = int(np.count_nonzero(result.final_necrosis >= .2 * p.carrying_capacity))
        records.append(row)
        print(f"{name}: {row['final']['cells']:.9e} viable cells; "
              f"{row['final']['volume_20_ml']:.3f} mL >=0.2K", flush=True)

    start = time.perf_counter()
    base = simulate(brain, p, compare=True)
    record("coupled", base, 2., p)
    record("fisher_kpp", base.comparison, 2., replace(p, model="fisher_kpp"))
    # Ensure that the manuscript reuses the previously reviewed demonstration.
    old = np.genfromtxt(ROOT / "outputs/coupled_example/growth_metrics.csv", delimiter=",", names=True)
    np.testing.assert_allclose(base.metrics["cells"], old["cells"], rtol=1e-7)
    for name, changes in [
        ("oxygen_only", dict(necrosis_rate=0.)),
        ("fixed_supply", dict(perfusion_suppression=0.)),
        ("zero_uptake", dict(oxygen_uptake_ratio=0.)),
        ("coupled_dt0125", dict(dt_days=.125)),
        ("coupled_dt00625", dict(dt_days=.0625)),
    ]:
        settings = replace(p, **changes)
        record(name, simulate(brain, settings), 2., settings)

    fine = results["coupled_dt00625"]
    for row in records:
        if row["name"] not in ("coupled", "coupled_dt0125"):
            continue
        result = results[row["name"]]
        errors = {}
        for field in ("density", "oxygen", "necrosis"):
            a = getattr(result, "final_" + field).astype(np.float64)
            b = getattr(fine, "final_" + field).astype(np.float64)
            delta = a - b
            errors[field] = dict(relative_l1=float(np.abs(delta).sum()/np.abs(b).sum()),
                                 relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(b)))
        row["errors_against_dt00625"] = errors

    reduction_error = np.max(np.abs(results["zero_uptake"].final_density.astype(np.float64)
                                    - base.comparison.final_density)) / p.carrying_capacity
    assert reduction_error < 1e-5, reduction_error
    native_p = replace(p, dt_days=.125)
    native = simulate(load_brainweb(1), native_p, compare=True)
    record("coupled_dx1", native, 1., native_p)
    record("fisher_kpp_dx1", native.comparison, 1., replace(native_p, model="fisher_kpp"))

    # A compact, lossless figure source; raw complete example outputs remain
    # available separately in outputs/coupled_example/.
    snapshots = dict(affine=base.affine, indices=np.asarray(base.indices),
                     K=np.array(p.carrying_capacity), spacing_mm=np.array(2.))
    for name, volume in [("viable", base.final_density), ("oxygen", base.final_oxygen),
                         ("necrosis", base.final_necrosis), ("reference", base.comparison.final_density),
                         ("mask", base.tissue_mask)]:
        for plane, data in get_planes(volume, base.indices).items():
            snapshots[f"{name}_{plane}"] = data
    for plane, data in base.backgrounds.items():
        snapshots[f"background_{plane}"] = data
    np.savez_compressed(OUT / "day90_planes.npz", **snapshots)

    lines = ["experiment,time_days," + ",".join(base.metrics)]
    for name, result in results.items():
        for i, day in enumerate(result.times):
            values = [str(float(result.metrics[key][i])) if key in result.metrics else ""
                      for key in base.metrics]
            lines.append(name + "," + str(float(day)) + "," + ",".join(values))
    (OUT / "timecourses.csv").write_text("\n".join(lines) + "\n")
    (OUT / "experiments.json").write_text(json.dumps(records, indent=2) + "\n")
    tracked = [ROOT / "src/brainweb_kpp/model.py", ROOT / "src/brainweb_kpp/data.py",
               ROOT / "src/brainweb_kpp/render.py", ROOT / "examples/coupled.json",
               ROOT / "uv.lock", HERE / "run_experiments.py"]
    provenance = dict(
        study="Deterministic anatomical-phantom study; all biological parameters assigned, not fitted",
        configuration=config, elapsed_seconds=time.perf_counter()-start,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in tracked},
        data_manifest=json.loads((DATA / "manifest.json").read_text()),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        working_tree_has_uncommitted_model_changes=True,
        zero_uptake_max_absolute_error_over_K=float(reduction_error),
        note="The Git revision alone does not identify the working sources; use the recorded source hashes.",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Completed {len(records)} model/step/grid cases in {provenance['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
