"""Generate manuscript tables, macros and vector figures from recovery records.

This does not refit the model. Run scripts/run_recovery_notebook.py to regenerate
the recorded experiment, including its source hashes, before rebuilding.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import nbformat
import numpy as np
from scipy.stats import chi2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECOVERY = ROOT / "outputs" / "synthetic_recovery"
FIG, GEN = HERE / "figures", HERE / "generated"


def rows_file(name, command, rows):
    (GEN / name).write_text("\\newcommand{\\" + command + "}{%\n" +
                            "\n".join(rows) + "\n}\n")


def scientific(value, signed=False):
    mantissa, exponent = format(value, "+.2e" if signed else ".2e").split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def main():
    summary = json.loads((RECOVERY / "summary.json").read_text())
    # Refuse stale sources: the notebook records these when the fit is executed.
    for name, expected in summary["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Recovery source changed: {name}; rerun the notebook")
    notebook = nbformat.read(ROOT / "synthetic_parameter_recovery.ipynb", as_version=4)
    code = "\n\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    if hashlib.sha256(code.encode()).hexdigest() != summary["notebook_code_sha256"]:
        raise ValueError("Recovery notebook code changed; rerun the notebook")
    if hashlib.sha256((RECOVERY / "recovery_data.npz").read_bytes()).hexdigest() != summary["recovery_data_sha256"]:
        raise ValueError("Recovery arrays do not match the recorded summary")
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    with np.load(RECOVERY / "recovery_data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    np.testing.assert_allclose(data["true_parameters"], summary["true_parameters"])
    np.testing.assert_allclose(np.quantile(data["bootstrap_parameters"], [.025, .975], axis=0).T,
                               summary["bootstrap_95_ci"])
    true, estimate = np.array(summary["true_parameters"]), np.array(summary["estimate"])
    local, bootstrap = np.array(summary["local_95_ci"]), np.array(summary["bootstrap_95_ci"])
    macros = {
        "RecoveryDTrue": f"{true[0]:.3f}", "RecoveryRhoTrue": f"{true[1]:.3f}",
        "RecoveryDEstimate": f"{estimate[0]:.4f}", "RecoveryRhoEstimate": f"{estimate[1]:.5f}",
        "RecoveryDRelativeError": f"{100*(estimate[0]/true[0]-1):+.2f}",
        "RecoveryRhoRelativeError": f"{100*(estimate[1]/true[1]-1):+.2f}",
        "RecoveryBootstraps": str(summary["bootstrap_replicates"]),
        "RecoveryTrials": str(summary["coverage_replicates"]),
        "RecoveryDCoverageCount": str(summary["coverage_counts"][0]),
        "RecoveryRhoCoverageCount": str(summary["coverage_counts"][1]),
        "RecoveryCorrelation": f"{summary['log_parameter_correlation']:+.3f}",
        "RecoveryDayNinetyRMSE": scientific(summary['withheld_clean_rmse'][-1]),
        "RecoveryDayNinetyGridRMSE": scientific(summary['coarse_fine_observation_rmse'][-1]),
        "RecoveryDayNinetyTimeRMSE": scientific(summary['halved_dt_observation_rmse'][-1]),
    }
    (GEN / "recovery_numbers.tex").write_text("% Generated from outputs/synthetic_recovery; do not edit.\n" +
        "\n".join(r"\newcommand{\%s}{%s}" % (key, value) for key, value in macros.items()) + "\n")
    names = [r"$D$ (mm$^2$/day)", r"$\rho$ (day$^{-1}$)"]
    rows_file("recovery_rows.tex", "RecoveryRows", [
        f"{name} & {true[k]:.5f} & {estimate[k]:.5f} & "
        f"[{local[k,0]:.5f}, {local[k,1]:.5f}] & [{bootstrap[k,0]:.5f}, {bootstrap[k,1]:.5f}] " + r"\\"
        for k, name in enumerate(names)])
    counts, fractions = summary["coverage_counts"], summary["coverage_fraction"]
    binomial = summary["coverage_binomial_95_ci"]
    rows_file("recovery_coverage_rows.tex", "RecoveryCoverageRows", [
        f"{name} & {scientific(summary['repeated_noise_bias'][k], signed=True)} & {scientific(summary['repeated_noise_rmse'][k])} & "
        f"{counts[k]}/{summary['coverage_replicates']} ({100*fractions[k]:.1f}\\%) & "
        f"[{100*binomial[k][0]:.1f}, {100*binomial[k][1]:.1f}]\\% " + r"\\"
        for k, name in enumerate(names)])
    rows_file("recovery_forecast_rows.tex", "RecoveryForecastRows", [
        f"{summary['scan_days'][j]:g} & {scientific(summary['withheld_clean_rmse'][k])} & "
        f"{summary['withheld_noisy_rmse'][k]:.5f} & "
        f"{100*summary['withheld_prediction_interval_voxel_coverage'][k]:.1f}\\% " + r"\\"
        for k, j in enumerate(summary["withheld_indices"])])

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), layout="constrained")
    ax = axes[0, 0]
    boot = data["bootstrap_parameters"]
    ax.scatter(boot[:, 0], boot[:, 1], s=9, alpha=.55, label="Bootstrap fits")
    # Delta-method covariance for display in physical units; inference uses logs.
    log_cov = np.array(summary["log_parameter_covariance"])
    covariance = np.diag(estimate) @ log_cov @ np.diag(estimate)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * np.sqrt(chi2.ppf(.95, 2) * values)
    ax.add_patch(Ellipse(estimate, width, height, angle=angle, fill=False,
                        edgecolor="#d95f02", linewidth=1.2, label="Local joint 95% (linearized)"))
    ax.scatter(*true, marker="*", s=65, c="black", label="Known")
    ax.scatter(*estimate, marker="+", s=45, c="#d95f02", label="Estimate")
    ax.set(title="a. Joint parameter uncertainty", xlabel=r"D (mm$^2$/day)", ylabel=r"$\rho$ (day$^{-1}$)")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    clean, predicted, observed = (data[key] for key in
                                ("clean_observations", "inferred_observations", "noisy_observations"))
    p = summary["inference_phantom"]
    coords = (np.arange(observed.shape[1]) + .5) * p["voxel_mm"] - p["width_mm"] / 2
    line = (slice(None), int(np.argmin(np.abs(coords-p["seed_center_mm"][1]))), observed.shape[-1]//2)
    mean_band, prediction_band = data["bootstrap_mean_band"], data["prediction_band"]
    ax.fill_between(coords, prediction_band[0, -1][line], prediction_band[1, -1][line],
                    color="#8dbbd3", alpha=.35, label="New-scan 95% prediction interval")
    ax.fill_between(coords, mean_band[0, -1][line], mean_band[1, -1][line],
                    color="#2978a0", alpha=.5, label="Mean-signal 95% confidence interval")
    ax.plot(coords, clean[-1][line], color="black", lw=1, label="Known noise-free signal")
    ax.plot(coords, predicted[-1][line], "--", color="#d95f02", lw=1, label="Inferred mean")
    ax.scatter(coords, observed[-1][line], c="black", s=9, label="Withheld scan")
    ax.set(title="b. Withheld day-90 line profile", xlabel="x (mm)",
           ylabel="Normalized signal", ylim=(-.08, .95))
    ax.legend(loc="upper right", fontsize=5.8)

    recovered, intervals = data["independent_recovered_parameters"], data["independent_local_intervals"]
    for k, ax in enumerate(axes[1]):
        order = np.argsort(recovered[:, k])
        for row, index in enumerate(order):
            inside = intervals[index, k, 0] <= true[k] <= intervals[index, k, 1]
            color = "#2978a0" if inside else "#c63d32"
            ax.plot(intervals[index, k], [row, row], color=color, lw=.7, alpha=.8)
        ax.axvline(true[k], color="black", ls="--", lw=1)
        ax.set(title=f"{'cd'[k]}. Local 95% intervals: {counts[k]}/{len(recovered)} cover truth",
               xlabel=[r"D (mm$^2$/day)", r"$\rho$ (day$^{-1}$)"][k], ylabel="Independent trial (sorted)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.locator_params(axis="x", nbins=4)
    fig.savefig(FIG / "synthetic_recovery.pdf")
    fig.savefig(FIG / "synthetic_recovery.png")
    plt.close(fig)
    print("Generated recovery vector figure, preview, macros and three tables from recorded arrays.")


if __name__ == "__main__":
    main()
