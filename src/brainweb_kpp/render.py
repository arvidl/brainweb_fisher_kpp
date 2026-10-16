"""Consistent notebook figures and 16:9 slide media. No system ffmpeg required."""
from pathlib import Path
from dataclasses import asdict
import importlib.metadata
import json
import platform

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from PIL import Image

from .model import Result

BG = "#101823"
FG = "#eff4fa"
MUTED = "#a6b6c9"


def display_plane(ax, background, density, capacity, *, spacing=1, threshold=0.02,
                  labels=("L", "R", "P", "A")):
    extent = (0, background.shape[0] * spacing, 0, background.shape[1] * spacing)
    ax.imshow(background.T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=extent)
    normed = density.T / capacity
    rgba = matplotlib.colormaps["inferno"](np.clip(normed, 0, 1))
    rgba[..., 3] = np.where(normed >= threshold, 0.82, 0)
    ax.imshow(rgba, origin="lower", extent=extent)
    ax.set_axis_off()
    for x, y, label in ((0.02, .5, labels[0]), (.98, .5, labels[1]),
                        (.5, .02, labels[2]), (.5, .98, labels[3])):
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                color=FG, fontsize=9, bbox=dict(facecolor=BG, edgecolor="none", alpha=.7, pad=1))


class SlideRenderer:
    def __init__(self, result, width=1920):
        if width < 640 or width % 32:
            raise ValueError("Width must be >=640 and divisible by 32 for 16:9 H.264")
        self.r = result
        self.fig = Figure(figsize=(width / 100, width * 9 / 16 / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        f = self.fig
        f.text(.035, .945, "Growth through brain tissue", color=FG, size=27, weight="bold")
        f.text(.035, .907, "3D FISHER–KPP  /  BRAINWEB NORMAL T1  /  ILLUSTRATIVE SIMULATION", color=MUTED, size=10)
        p = result.parameters
        f.text(.035, .866,
               f"D white / gray: {p['d_white']:g} / {p['d_gray']:g} mm²/day     "
               f"ρ: {p['rho']:g} day⁻¹     K: {p['carrying_capacity']:,.0f} cells/mm³     "
               f"Δx: {result.metadata['spacing_mm']:g} mm", color=MUTED, size=11)
        self.day = f.text(.965, .94, "", color="#ffcf6a", size=27, ha="right", weight="bold")
        self.images = []
        orientations = {"Axial": ("L", "R", "P", "A"), "Coronal": ("L", "R", "I", "S"),
                        "Sagittal": ("P", "A", "I", "S")}
        xyz = nib.affines.apply_affine(result.affine, result.indices)
        for j, name in enumerate(("Axial", "Coronal", "Sagittal")):
            ax = f.add_axes([.035 + j * .305, .325, .275, .485], facecolor=BG)
            display_plane(ax, result.backgrounds[name], result.planes[name][0], p["carrying_capacity"],
                          spacing=result.metadata["spacing_mm"], labels=orientations[name])
            # MRI and density are the only two AxesImage instances.
            self.images.append(ax.images[1])
            coordinate = xyz[(2, 1, 0)[j]]
            axis_name = ("z", "y", "x")[j]
            ax.set_title(f"{name}  ·  {axis_name} = {coordinate:g} mm", color=FG, size=12, pad=12)
        cax = f.add_axes([.958, .39, .010, .34])
        f.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
                   cax=cax, ticks=[0, .2, .5, .8, 1])
        cax.tick_params(colors=MUTED, labelsize=8)
        cax.set_title("n / K", color=FG, size=10, pad=10)
        ax = f.add_axes([.065, .105, .54, .15], facecolor=BG)
        for key, color, label in (("volume_20_ml", "#ffcf6a", "n ≥ 0.2 K"),
                                  ("volume_80_ml", "#ff755f", "n ≥ 0.8 K")):
            ax.plot(result.times, result.metrics[key], color=color, label=label, lw=1.5, alpha=.65)
        self.cursor = ax.axvline(0, color=FG, lw=1)
        ax.set(xlabel="Time (days)", ylabel="Volume (mL)", xlim=(0, result.times[-1]))
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values(): spine.set_color("#364253")
        ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
        ax.grid(alpha=.10, color=FG)
        self.stats = f.text(.68, .17, "", color=FG, size=15, linespacing=1.8)
        f.text(.035, .026, "Fixed anatomy · isotropic tissue-dependent diffusion · no-flux boundaries · no treatment or mass effect",
               color=MUTED, size=9)
        f.text(.965, .026, "BrainWeb / MNI, McGill", color=MUTED, size=9, ha="right")

    def frame(self, i):
        p = self.r.parameters
        for image, name in zip(self.images, ("Axial", "Coronal", "Sagittal")):
            u = self.r.planes[name][i].T / p["carrying_capacity"]
            rgba = matplotlib.colormaps["inferno"](np.clip(u, 0, 1))
            rgba[..., 3] = np.where(u >= .02, .82, 0)
            image.set_data(rgba)
        self.day.set_text(f"DAY {self.r.times[i]:05.1f}")
        self.cursor.set_xdata([self.r.times[i]] * 2)
        m = self.r.metrics
        self.stats.set_text(f"Infiltrated volume  {m['volume_20_ml'][i]:6.1f} mL\n"
                            f"Dense core             {m['volume_80_ml'][i]:6.1f} mL\n"
                            f"Total cells               {m['cells'][i]:.2e}")
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


def save_result(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "parameters.json").write_text(json.dumps(result.parameters, indent=2) + "\n")
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "numba", "nibabel", "matplotlib", "imageio", "pillow")}
    metadata = dict(result.metadata, python=platform.python_version(), platform=platform.platform(),
                    versions=versions, coordinates="RAS+ mm, neurological display", indices=list(map(int, result.indices)))
    from .data import DATA
    if (DATA / "manifest.json").exists():
        metadata["data_manifest"] = json.loads((DATA / "manifest.json").read_text())
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    arrays = {"times": result.times, "affine": result.affine,
              "parameters": json.dumps(result.parameters), "metadata": json.dumps(metadata)}
    arrays.update({"plane_" + key: value for key, value in result.planes.items()})
    arrays.update({"background_" + key: value for key, value in result.backgrounds.items()})
    arrays.update({"metric_" + key: value for key, value in result.metrics.items()})
    np.savez_compressed(output / "simulation.npz", **arrays)
    nib.save(nib.Nifti1Image(result.final_density, result.affine), output / "final_density_cells_mm3.nii.gz")
    header = "time_days," + ",".join(result.metrics)
    np.savetxt(output / "growth_metrics.csv", np.column_stack([result.times, *result.metrics.values()]),
               delimiter=",", header=header, comments="")


def load_result(output):
    output = Path(output)
    with np.load(output / "simulation.npz", allow_pickle=False) as a:
        p, meta = json.loads(str(a["parameters"])), json.loads(str(a["metadata"]))
        return Result(a["times"], {key[6:]: a[key] for key in a if key.startswith("plane_")},
                      {key[11:]: a[key] for key in a if key.startswith("background_")},
                      np.asarray(nib.load(output / "final_density_cells_mm3.nii.gz").dataobj),
                      {key[7:]: a[key] for key in a if key.startswith("metric_")},
                      p, meta, a["affine"], tuple(meta["indices"]))


def export_media(result, output, *, fps=12, width=1920, formats=("mp4", "gif"), progress=None):
    if not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if set(formats) - {"mp4", "gif"}:
        raise ValueError("Formats must be mp4 and/or gif")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    renderer = SlideRenderer(result, width)
    Image.fromarray(renderer.frame(len(result.times) - 1)).save(output / "poster.png")
    # H.264 + yuv420p + faststart, compatible with common slide software.
    writer = None
    if "mp4" in formats:
        writer = imageio.get_writer(output / "tumor_growth.mp4", fps=fps, codec="libx264",
                                    pixelformat="yuv420p",
                                    quality=8, macro_block_size=1,
                                    output_params=["-movflags", "+faststart"])
    gif_frames = []
    # One shared palette across the animation prevents per-frame palette flicker.
    gif_width = min(width, 960)
    gif_height = gif_width * 9 // 16
    palette_samples = [Image.fromarray(renderer.frame(i)).resize((gif_width, gif_height))
                       for i in np.linspace(0, len(result.times) - 1, 4).astype(int)] if "gif" in formats else []
    palette = None
    if palette_samples:
        montage = Image.new("RGB", (gif_width, gif_height * 4))
        for j, sample in enumerate(palette_samples): montage.paste(sample, (0, j * gif_height))
        palette = montage.quantize(colors=256, dither=Image.Dither.NONE)
    order = [0] * max(1, fps // 2) + list(range(len(result.times))) + [len(result.times) - 1] * fps
    try:
        for j, i in enumerate(order):
            frame = renderer.frame(i)
            if writer: writer.append_data(frame)
            if palette is not None:
                gif_frames.append(Image.fromarray(frame).resize((gif_width, gif_height), Image.Resampling.LANCZOS)
                                  .quantize(palette=palette, dither=Image.Dither.NONE))
            if progress: progress(j + 1, len(order))
    finally:
        if writer: writer.close()
    if gif_frames:
        gif_frames[0].save(output / "tumor_growth.gif", save_all=True, append_images=gif_frames[1:],
                           duration=round(1000 / fps / 10) * 10, loop=0, optimize=False, disposal=1)
    media = {"fps": fps, "width": width, "height": width * 9 // 16,
             "gif_width": gif_width, "display_threshold_n_over_K": .02,
             "simulation_days": float(result.times[-1]), "video_seconds": len(order) / fps,
             "formats": list(formats), "codec": "H.264", "pixel_format": "yuv420p"}
    if gif_frames:
        with Image.open(output / "tumor_growth.gif") as gif:
            durations = []
            for i in range(gif.n_frames):
                gif.seek(i)
                durations.append(gif.info.get("duration", 0))
            media.update(gif_frames=gif.n_frames, gif_seconds=sum(durations) / 1000)
    (output / "media_metadata.json").write_text(json.dumps(media, indent=2) + "\n")
    return output
