"""Build BrainWeb inference tables and vector figures from verified records.

No parameters are refitted here. Notebook Part B produces the source records.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import nbformat
import numpy as np
from scipy.stats import binomtest

from brainweb_kpp.data import load_brainweb, verify_data
from make_recovery_figures import rows_file, scientific

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDS = ROOT / "outputs" / "brainweb_recovery"
FIG, GEN = HERE / "figures", HERE / "generated"


def load_records():
    """Reject stale code/data and recompute reported sampling/forecast metrics."""
    summary = json.loads((RECORDS / "summary.json").read_text())
    for name, expected in summary["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"BrainWeb recovery source changed: {name}; rerun Part B")
    notebook = nbformat.read(ROOT / "synthetic_parameter_recovery.ipynb", as_version=4)
    code = "\n\n".join(c.source for c in notebook.cells if c.cell_type == "code"
                      and c.metadata.get("recovery_experiment") == summary["notebook_code_scope"])
    if hashlib.sha256(code.encode()).hexdigest() != summary["notebook_code_sha256"]:
        raise ValueError("BrainWeb notebook code changed; rerun Part B")
    archive = RECORDS / "recovery_data.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != summary["recovery_data_sha256"]:
        raise ValueError("BrainWeb recovery arrays do not match the summary")
    if verify_data() != summary["brainweb_manifest"]:
        raise ValueError("BrainWeb manifest changed")
    with np.load(archive, allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    true = np.array(summary["true_parameters"])
    np.testing.assert_array_equal(data["true_parameters"], true)
    np.testing.assert_allclose(np.quantile(data["bootstrap_parameters"], [.025, .975], axis=0).T,
                               summary["bootstrap_95_ci"])
    recovered, ci = data["independent_recovered_parameters"], data["independent_local_intervals"]
    contains = (ci[:, :, 0] <= true) & (true <= ci[:, :, 1])
    np.testing.assert_array_equal(contains.sum(axis=0), summary["coverage_counts"])
    np.testing.assert_allclose(contains.mean(axis=0), summary["coverage_fraction"])
    np.testing.assert_allclose(recovered.mean(axis=0) - true, summary["repeated_noise_bias"])
    np.testing.assert_allclose(np.sqrt(np.mean((recovered-true)**2, axis=0)), summary["repeated_noise_rmse"])
    np.testing.assert_allclose([tuple(binomtest(int(k), len(recovered)).proportion_ci(method="exact"))
                               for k in contains.sum(axis=0)], summary["coverage_binomial_95_ci"])
    assert len(recovered) == summary["coverage_replicates"]
    assert len(data["bootstrap_parameters"]) == summary["bootstrap_replicates"]
    for region, mask in [("whole_FOV", np.ones(data["near_seed_region"].shape, bool)),
                         ("near_seed_18mm", data["near_seed_region"])]:
        predicted, clean, observed = (data[k][:, mask] for k in
            ("inferred_observations", "clean_observations", "noisy_observations"))
        metrics = summary["forecast_metrics"][region]
        np.testing.assert_allclose(np.sqrt(np.mean((predicted-clean)**2, axis=1)), metrics["clean_rmse"])
        np.testing.assert_allclose(np.sqrt(np.mean((predicted-observed)**2, axis=1)), metrics["noisy_rmse"])
        lower, upper = data["prediction_band"][:, :, mask]
        np.testing.assert_allclose(np.mean((lower <= observed) & (observed <= upper), axis=1),
                                   metrics["prediction_voxel_fraction"])
    for source, key in [("coarse_at_truth", "temporal_rmse_dt_0.5_vs_0.125"),
                        ("half_dt_at_truth", "temporal_rmse_dt_0.25_vs_0.125"),
                        ("fine_2mm_clean_observations", "anatomy_grid_observation_rmse_3mm_vs_2mm")]:
        np.testing.assert_allclose(np.sqrt(np.mean((data[source]-data["clean_observations"])**2,
                                                  axis=(1, 2, 3))), summary[key])
    np.testing.assert_allclose(data["fine_2mm_noisy_observations"]-data["fine_2mm_clean_observations"],
                               data["noisy_observations"]-data["clean_observations"], atol=1e-15)
    return summary, data


def tables(s):
    true, estimate = np.array(s["true_parameters"]), np.array(s["estimate"])
    local, boot = np.array(s["local_95_ci"]), np.array(s["bootstrap_95_ci"])
    stress = np.array(s["anatomy_stress_noiseless_estimate"])
    macros = {
        "BWBootstraps": str(s["bootstrap_replicates"]), "BWTrials": str(s["coverage_replicates"]),
        "BWWhiteEstimate": f"{estimate[0]:.5f}", "BWGrayEstimate": f"{estimate[1]:.5f}",
        "BWRhoEstimate": f"{estimate[2]:.5f}",
        "BWWhiteError": f"{100*(estimate[0]/true[0]-1):+.2f}",
        "BWGrayError": f"{100*(estimate[1]/true[1]-1):+.2f}",
        "BWRhoError": f"{100*(estimate[2]/true[2]-1):+.2f}",
        "BWWhiteCoverage": str(s["coverage_counts"][0]), "BWGrayCoverage": str(s["coverage_counts"][1]),
        "BWRhoCoverage": str(s["coverage_counts"][2]),
        "BWCorrelation": f"{s['log_parameter_correlation'][0][1]:+.3f}",
        "BWCondition": f"{s['weighted_jacobian_singular_values'][0]/s['weighted_jacobian_singular_values'][-1]:.2f}",
        "BWDayNinetyRMSE": scientific(s["forecast_metrics"]["whole_FOV"]["clean_rmse"][-1]),
        "BWNearDayNinetyRMSE": scientific(s["forecast_metrics"]["near_seed_18mm"]["clean_rmse"][-1]),
        "BWDayNinetyGridRMSE": scientific(s["anatomy_grid_observation_rmse_3mm_vs_2mm"][-1]),
        "BWDayNinetyTimeRMSE": scientific(s["temporal_rmse_dt_0.5_vs_0.125"][-1]),
        "BWHalfTimeRMSE": scientific(s["temporal_rmse_dt_0.25_vs_0.125"][-1]),
        "BWGrayStressEstimate": f"{stress[1]:.6f}",
        "BWGrayStressShift": f"{100*(stress[1]/true[1]-1):.2f}",
    }
    (GEN / "brainweb_recovery_numbers.tex").write_text("% Generated from outputs/brainweb_recovery; do not edit.\n" +
        "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in macros.items()) + "\n")
    names = [r"$D_{\rm WM}$ (mm$^2$/day)", r"$D_{\rm GM}$ (mm$^2$/day)", r"$\rho$ (day$^{-1}$)"]
    rows_file("brainweb_recovery_rows.tex", "BWRecoveryRows", [
        f"{name} & {true[k]:.5f} & {estimate[k]:.5f} & [{local[k,0]:.5f}, {local[k,1]:.5f}] & "
        f"[{boot[k,0]:.5f}, {boot[k,1]:.5f}] " + r"\\" for k, name in enumerate(names)])
    rows_file("brainweb_coverage_rows.tex", "BWCoverageRows", [
        f"{name} & {scientific(s['repeated_noise_bias'][k], signed=True)} & "
        f"{scientific(s['repeated_noise_rmse'][k])} & {s['coverage_counts'][k]}/{s['coverage_replicates']} "
        f"({100*s['coverage_fraction'][k]:.1f}\\%) & "
        f"[{100*s['coverage_binomial_95_ci'][k][0]:.1f}, {100*s['coverage_binomial_95_ci'][k][1]:.1f}]\\% "
        + r"\\" for k, name in enumerate(names)])
    rows_file("brainweb_forecast_rows.tex", "BWForecastRows", [
        f"{s['scan_days'][j]:g} & {label} & {metrics['clean_rmse'][j]:.5f} & {metrics['noisy_rmse'][j]:.5f} & "
        f"{100*metrics['prediction_voxel_fraction'][j]:.1f}\\% " + r"\\"
        for j in s["withheld_indices"] for region, label in [("whole_FOV", "Full FOV"), ("near_seed_18mm", "Near seed")]
        for metrics in [s["forecast_metrics"][region]]])
    rows_file("brainweb_anatomy_rows.tex", "BWAnatomyRows", [
        f"{name} & {true[k]:.5f} & {stress[k]:.6f} & {100*(stress[k]/true[k]-1):+.2f}\\% & "
        f"{s['anatomy_stress_noisy_estimate'][k]:.6f} " + r"\\" for k, name in enumerate(names)])


def recovery_figure(s, d):
    true, estimate = np.array(s["true_parameters"]), np.array(s["estimate"])
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 5.4), layout="constrained")
    ax = axes[0, 0]
    boot = d["bootstrap_parameters"]
    ax.scatter(boot[:, 0], boot[:, 1], s=8, alpha=.6, label="Bootstrap fits")
    ax.scatter(*true[:2], marker="*", s=60, color="black", label="Known")
    ax.scatter(*estimate[:2], marker="+", s=40, color="#d95f02", label="Estimate")
    ax.set(title="a. Tissue-rate dependence", xlabel=r"$D_{\rm WM}$ (mm$^2$/day)", ylabel=r"$D_{\rm GM}$ (mm$^2$/day)")
    ax.legend(fontsize=5.5, loc="lower left")
    ax = axes[0, 1]
    ax.hist(boot[:, 2], bins=14, color="#9b4d96", alpha=.75)
    ax.axvspan(*s["bootstrap_95_ci"][2], color="gray", alpha=.13, label="Bootstrap 95% CI")
    ax.axvline(true[2], color="black", label="Known")
    ax.axvline(estimate[2], color="#d95f02", ls="--", label="Estimate")
    ax.set(title="b. Proliferation uncertainty", xlabel=r"$\rho$ (day$^{-1}$)", ylabel="Bootstrap fits")
    ax.legend(fontsize=5.3, loc="upper left")
    ax = axes[0, 2]
    centers = d["observation_centers_ras_mm"]
    y = int(np.argmin(abs(centers[0, :, 0, 1]-s["design"]["seed_center_mm"][1])))
    z = int(np.argmin(abs(centers[0, 0, :, 2]-s["design"]["seed_center_mm"][2])))
    line, x = (slice(None), y, z), centers[:, 0, 0, 0]
    mean, pred = d["bootstrap_mean_band"], d["prediction_band"]
    ax.fill_between(x, pred[0, -1][line], pred[1, -1][line], color="#8dbbd3", alpha=.35, label="New-scan 95% PI")
    ax.fill_between(x, mean[0, -1][line], mean[1, -1][line], color="#2978a0", alpha=.5, label="Mean-signal 95% CI")
    ax.plot(x, d["clean_observations"][-1][line], color="black", lw=.9, label="Known mean")
    ax.plot(x, d["inferred_observations"][-1][line], "--", color="#d95f02", lw=.9, label="Fitted mean")
    ax.scatter(x, d["noisy_observations"][-1][line], color="black", s=8, label="Withheld scan")
    ax.set(title="c. Withheld day-90 profile", xlabel="RAS x (mm)", ylabel="Normalized signal", ylim=(-.08, .90))
    ax.legend(fontsize=5.0, loc="upper right")
    for k, ax in enumerate(axes[1]):
        recovered, ci = d["independent_recovered_parameters"], d["independent_local_intervals"]
        for row, index in enumerate(np.argsort(recovered[:, k])):
            inside = ci[index, k, 0] <= true[k] <= ci[index, k, 1]
            ax.plot(ci[index, k], [row, row], color="#2978a0" if inside else "#c63d32", lw=.7, alpha=.85)
        ax.axvline(true[k], color="black", ls="--", lw=.9)
        ax.set(title=f"{'def'[k]}. Local CI coverage: {s['coverage_counts'][k]}/{len(recovered)}",
               xlabel=[r"$D_{\rm WM}$ (mm$^2$/day)", r"$D_{\rm GM}$ (mm$^2$/day)", r"$\rho$ (day$^{-1}$)"][k],
               ylabel="Independent trial (sorted)")
    for ax in axes.flat:
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.locator_params(axis="x", nbins=3)
    fig.savefig(FIG / "brainweb_recovery.pdf")
    fig.savefig(FIG / "brainweb_recovery.png")
    plt.close(fig)


def anatomy_figure(s, d):
    brain = load_brainweb(s["inverse_grid"]["spacing_mm"])
    z = brain.voxel(s["design"]["seed_center_mm"])[2]
    classes = np.where(brain.labels == 3, 2, np.where(np.isin(brain.labels, [2, 8]), 1, 0))
    h = brain.spacing
    extent = [brain.affine[0, 3]-h/2, brain.affine[0, 3]+(brain.labels.shape[0]-.5)*h,
              brain.affine[1, 3]-h/2, brain.affine[1, 3]+(brain.labels.shape[1]-.5)*h]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), layout="constrained")
    ax = axes[0, 0]
    im = ax.imshow(classes[:, :, z].T, origin="lower", extent=extent,
                   cmap=ListedColormap(["#19232d", "#c6a99b", "#e7e8eb"]), vmin=-.5, vmax=2.5)
    center = s["design"]["observation_center_mm"]
    ax.add_patch(Rectangle((center[0]-30, center[1]-30), 60, 60, fill=False, color="#edb74d", lw=1.1))
    ax.plot(*s["design"]["seed_center_mm"][:2], "+", color="#d34a48", ms=6)
    cb = fig.colorbar(im, ax=ax, shrink=.8, ticks=[0, 1, 2], pad=.02)
    cb.ax.set_yticklabels(["Excluded", "GM/glial", "WM"])
    ax.set(title="a. Known 3 mm anatomy, z = 30 mm", xlabel="RAS x (mm)", ylabel="RAS y (mm)")
    ax = axes[0, 1]
    centers = d["observation_centers_ras_mm"]
    obs_z = int(np.argmin(abs(centers[0, 0, :, 2]-s["design"]["seed_center_mm"][2])))
    difference = (d["fine_2mm_clean_observations"][-1]-d["clean_observations"][-1])[:, :, obs_z]
    extent = [centers[0, 0, 0, 0]-3, centers[-1, 0, 0, 0]+3,
              centers[0, 0, 0, 1]-3, centers[0, -1, 0, 1]+3]
    limit = max(abs(difference.min()), abs(difference.max()))
    im = ax.imshow(difference.T, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=ax, shrink=.8, pad=.02, label="Signal difference")
    ax.set(title="b. Day 90: 2 mm minus 3 mm signal", xlabel="RAS x (mm)", ylabel="RAS y (mm)")
    ax = axes[1, 0]
    shift = 100*(np.array(s["anatomy_stress_noiseless_estimate"])/s["true_parameters"]-1)
    bars = ax.bar([r"$D_{\rm WM}$", r"$D_{\rm GM}$", r"$\rho$"], shift, color=["#2978a0", "#5e995b", "#9b4d96"])
    ax.axhline(0, color="black", lw=.7)
    for bar, value in zip(bars, shift):
        ax.annotate(f"{value:+.2f}%", (bar.get_x()+bar.get_width()/2, max(0, value)),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ax.set(title="c. Fitting noiseless 2 mm data at 3 mm", ylabel="Rate shift from known value (%)", ylim=(-2.5, 18))
    ax = axes[1, 1]
    ax.semilogy(s["scan_days"], s["anatomy_grid_observation_rmse_3mm_vs_2mm"], "o-", ms=3,
                label="2 mm vs 3 mm anatomy/grid")
    ax.semilogy(s["scan_days"], s["temporal_rmse_dt_0.5_vs_0.125"], "s-", ms=3,
                label="dt 0.5 vs 0.125 day (3 mm)")
    ax.axhline(s["noise_sd"], color="gray", ls="--", lw=.9, label="Noise SD = 0.02")
    ax.set(title="d. Observation-space discrepancy", xlabel="Day", ylabel="Signal RMSE (log scale)")
    ax.legend(fontsize=6, loc="upper left")
    fig.savefig(FIG / "brainweb_anatomy_sensitivity.pdf")
    fig.savefig(FIG / "brainweb_anatomy_sensitivity.png")
    plt.close(fig)


def main():
    s, d = load_records()
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "savefig.dpi": 240,
                         "axes.spines.top": False, "axes.spines.right": False})
    tables(s)
    recovery_figure(s, d)
    anatomy_figure(s, d)
    print("Verified BrainWeb source/code/data hashes and array metrics; generated two vector figures and four tables.")


if __name__ == "__main__":
    main()
