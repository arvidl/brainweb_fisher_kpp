"""Consistent notebook figures and 16:9 slide media. No system ffmpeg required."""
from pathlib import Path
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

from .model import Result, get_planes

BG = "#101823"
FG = "#eff4fa"
MUTED = "#a6b6c9"

VIEW_STYLES = {
    "viable": ("Viable tumor", "c / K", "inferno"),
    "oxygen": ("Oxygen availability", "o / normal baseline", "viridis"),
    "necrosis": ("Necrotic tumor material", "z / K", "magma"),
    "reference": ("Fisher–KPP reference", "c / K", "inferno"),
}


def field_data(result, view="viable"):
    """Select synchronized planes/volumes without changing model parameters."""
    if view not in VIEW_STYLES:
        raise ValueError(f"Unknown field view: {view}")
    source = result.comparison if view == "reference" else result
    if source is None:
        raise ValueError("Run with a Fisher–KPP reference to select that view")
    if view == "oxygen":
        planes, volumes, scale = source.oxygen_planes, source.oxygen_volumes, 1.
    elif view == "necrosis":
        planes, volumes, scale = source.necrosis_planes, source.necrosis_volumes, source.parameters["carrying_capacity"]
    else:
        planes, volumes, scale = source.planes, source.volumes, source.parameters["carrying_capacity"]
    if planes is None:
        raise ValueError(f"The {view} field is available only for the coupled model")
    return source, planes, volumes, scale


def overlay_rgba(density, capacity, *, cmap="inferno", threshold=.02, mask=None):
    normed = density.T / capacity
    rgba = matplotlib.colormaps[cmap](np.clip(normed, 0, 1))
    visible = np.isfinite(normed) & (normed >= threshold)
    if mask is not None:
        visible &= mask.T
    rgba[..., 3] = np.where(visible, .82, 0)
    return rgba


def display_plane(ax, background, density, capacity, *, spacing=1, threshold=0.02,
                  labels=("L", "R", "P", "A"), cmap="inferno", mask=None):
    extent = (0, background.shape[0] * spacing, 0, background.shape[1] * spacing)
    ax.imshow(background.T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=extent)
    rgba = overlay_rgba(density, capacity, cmap=cmap, threshold=threshold, mask=mask)
    ax.imshow(rgba, origin="lower", extent=extent)
    ax.set_axis_off()
    for x, y, label in ((0.02, .5, labels[0]), (.98, .5, labels[1]),
                        (.5, .02, labels[2]), (.5, .98, labels[3])):
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                color=FG, fontsize=9, bbox=dict(facecolor=BG, edgecolor="none", alpha=.7, pad=1))


class SlideRenderer:
    def __init__(self, result, width=1920, view="viable"):
        if width < 640 or width % 32:
            raise ValueError("Width must be >=640 and divisible by 32 for 16:9 H.264")
        self.r = result
        self.view = view
        self.source, self.planes, _, self.scale = field_data(result, view)
        title, label, self.cmap = VIEW_STYLES[view]
        from .model import get_planes
        self.masks = get_planes(result.tissue_mask, result.indices) if view == "oxygen" else None
        self.fig = Figure(figsize=(width / 100, width * 9 / 16 / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        f = self.fig
        f.text(.035, .945, title, color=FG, size=27, weight="bold")
        model_name = "COUPLED TUMOR / OXYGEN / NECROSIS" if result.final_oxygen is not None else "FISHER–KPP"
        f.text(.035, .907, f"3D {model_name}  ·  BRAINWEB NORMAL T1  ·  ILLUSTRATIVE", color=MUTED, size=10)
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
            display_plane(ax, result.backgrounds[name], self.planes[name][0], self.scale,
                          spacing=result.metadata["spacing_mm"], labels=orientations[name], cmap=self.cmap,
                          threshold=0 if view == "oxygen" else .02,
                          mask=self.masks[name] if self.masks is not None else None)
            # MRI and density are the only two AxesImage instances.
            self.images.append(ax.images[1])
            coordinate = xyz[(2, 1, 0)[j]]
            axis_name = ("z", "y", "x")[j]
            ax.set_title(f"{name}  ·  {axis_name} = {coordinate:g} mm", color=FG, size=12, pad=12)
        cax = f.add_axes([.958, .39, .010, .34])
        f.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=self.cmap),
                   cax=cax, ticks=[0, .2, .5, .8, 1])
        cax.tick_params(colors=MUTED, labelsize=8)
        cax.set_title("o" if view == "oxygen" else label, color=FG, size=10, pad=10)
        ax = f.add_axes([.065, .105, .54, .15], facecolor=BG)
        for key, color, label in (("volume_20_ml", "#ffcf6a", "Viable ≥ 0.2 K"),
                                  ("volume_80_ml", "#ff755f", "Viable ≥ 0.8 K")):
            ax.plot(result.times, result.metrics[key], color=color, label=label, lw=1.5, alpha=.65)
        if result.comparison is not None:
            ax.plot(result.comparison.times, result.comparison.metrics["volume_20_ml"], color="#68ceef",
                    label="Fisher–KPP ≥ 0.2 K", lw=1.5, ls="--")
        if result.final_necrosis is not None:
            ax.plot(result.times, result.metrics["necrotic_volume_20_ml"], color="#d98be8",
                    label="Necrotic ≥ 0.2 K", lw=1.5)
        self.cursor = ax.axvline(0, color=FG, lw=1)
        ax.set(xlabel="Time (days)", ylabel="Volume (mL)", xlim=(0, result.times[-1]))
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values(): spine.set_color("#364253")
        ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
        ax.grid(alpha=.10, color=FG)
        self.stats = f.text(.68, .12, "", color=FG, size=14, linespacing=1.6)
        f.text(.035, .026, "Fixed anatomy · isotropic tissue-dependent diffusion · no-flux boundaries · no treatment or mass effect",
               color=MUTED, size=9)
        f.text(.965, .026, "BrainWeb / MNI, McGill", color=MUTED, size=9, ha="right")
        for text in f.findobj(matplotlib.text.Text):
            text.set_fontsize(text.get_fontsize() * width / 1920)

    def frame(self, i):
        for image, name in zip(self.images, ("Axial", "Coronal", "Sagittal")):
            rgba = overlay_rgba(self.planes[name][i], self.scale, cmap=self.cmap,
                                threshold=0 if self.view == "oxygen" else .02,
                                mask=self.masks[name] if self.masks is not None else None)
            image.set_data(rgba)
        self.day.set_text(f"DAY {self.r.times[i]:05.1f}")
        self.cursor.set_xdata([self.r.times[i]] * 2)
        m = self.r.metrics
        model_label = "Coupled" if self.r.final_oxygen is not None else "Fisher–KPP"
        stats = (f"{model_label} ≥ 0.2 K  {m['volume_20_ml'][i]:6.1f} mL\n"
                 f"{model_label} cells       {m['cells'][i]:.2e}")
        if self.r.final_necrosis is not None:
            stats += (f"\nNecrotic ≥ 0.2 K  {m['necrotic_volume_20_ml'][i]:6.1f} mL"
                      f"\nMinimum oxygen  {m['min_oxygen'][i]:.3f}")
        if self.r.comparison is not None:
            stats += f"\nReference cells   {self.r.comparison.metrics['cells'][i]:.2e}"
        self.stats.set_text(stats)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


class ComparisonRenderer:
    """Four fields, one frame index and one set of slices for the whole movie."""
    fields = ("viable", "oxygen", "necrosis", "reference")

    def __init__(self, result, width=1920, *, plane="all", indices=None, brain=None):
        if width < 640 or width % 32:
            raise ValueError("Width must be >=640 and divisible by 32")
        if plane not in ("all", "Axial", "Coronal", "Sagittal"):
            raise ValueError("Plane must be all, Axial, Coronal or Sagittal")
        reference = result.comparison
        if result.final_oxygen is None or result.final_necrosis is None or reference is None:
            raise ValueError("Side-by-side export requires coupled fields and a Fisher–KPP reference")
        if not np.array_equal(result.times, reference.times):
            raise ValueError("Comparison fields must have identical saved times")
        if (not np.array_equal(result.affine, reference.affine)
                or result.final_density.shape != reference.final_density.shape
                or tuple(result.indices) != tuple(reference.indices)):
            raise ValueError("Comparison fields must share the same grid and saved slices")
        self.r = result
        self.indices = tuple(result.indices if indices is None else indices)
        shape = result.final_density.shape
        if (len(self.indices) != 3 or any(int(i) != i or i < 0 or i >= n
                                          for i, n in zip(self.indices, shape))):
            raise ValueError("Slice indices must be three integers inside the MRI grid")
        self.indices = tuple(map(int, self.indices))
        self.names = ("Axial", "Coronal", "Sagittal") if plane == "all" else (plane,)
        self.xyz = nib.affines.apply_affine(result.affine, self.indices)
        self.data = {view: field_data(result, view) for view in self.fields}
        self.custom_slices = self.indices != tuple(result.indices)
        if self.custom_slices:
            if brain is None or any(data[2] is None for data in self.data.values()):
                raise ValueError("Other slices require the anatomy and saved 3D frames for every field")
            if brain.t1.shape != shape or not np.array_equal(brain.affine, result.affine):
                raise ValueError("Export anatomy must match the simulation grid")
            backgrounds = get_planes(brain.t1, self.indices)
        else:
            backgrounds = result.backgrounds
        for _, histories, volumes, _ in self.data.values():
            if any(len(histories[name]) != len(result.times) for name in self.names):
                raise ValueError("Comparison fields must have one plane per saved time")
            if self.custom_slices and len(volumes) != len(result.times):
                raise ValueError("Comparison volumes must have one volume per saved time")
        self.masks = get_planes(result.tissue_mask, self.indices)
        # Three anatomical rows retain the explorer's 4:3 overview; one row uses 16:9.
        height = width * (3 / 4 if plane == "all" else 9 / 16)
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        f = self.fig
        f.text(.025, .96, "Tumor–brain growth · side-by-side comparison", color=FG, size=25, weight="bold")
        f.text(.025, .923, f"COUPLED MODEL + MATCHED FISHER–KPP  ·  Δx {result.metadata['spacing_mm']:g} mm  ·  ILLUSTRATIVE",
               color=MUTED, size=11)
        self.day = f.text(.975, .96, "", color="#ffcf6a", size=25, ha="right", weight="bold")
        self.images, self.stats, self.axes, self.colorbars = {}, {}, {}, {}
        orientations = {"Axial": ("L", "R", "P", "A"), "Coronal": ("L", "R", "I", "S"),
                        "Sagittal": ("P", "A", "I", "S")}
        rows = len(self.names)
        row_height = (.68 - .024 * (rows - 1)) / rows
        for col, view in enumerate(self.fields):
            left, col_width = .025 + col * .2425, .2225
            title, label, cmap = VIEW_STYLES[view]
            f.text(left + col_width / 2, .865, title, color=FG, size=17, ha="center", weight="bold")
            self.stats[view] = f.text(left + col_width / 2, .832, "", color=MUTED, size=11, ha="center")
            self.images[view], self.axes[view] = {}, {}
            planes = self._planes(view, 0)
            for row, name in enumerate(self.names):
                bottom = .78 - (row + 1) * row_height - row * .024
                ax = f.add_axes([left, bottom, col_width, row_height], facecolor=BG)
                display_plane(ax, backgrounds[name], planes[name], self.data[view][3],
                              spacing=result.metadata["spacing_mm"], labels=orientations[name], cmap=cmap,
                              threshold=0 if view == "oxygen" else .02,
                              mask=self.masks[name] if view == "oxygen" else None)
                axis, coord = {"Axial": ("z", self.xyz[2]), "Coronal": ("y", self.xyz[1]),
                               "Sagittal": ("x", self.xyz[0])}[name]
                ax.set_title(f"{name} · {axis} = {coord:g} mm", color=MUTED, size=11, pad=5)
                self.axes[view][name], self.images[view][name] = ax, ax.images[1]
            cax = f.add_axes([left + .01, .06, col_width - .02, .008])
            self.colorbars[view] = f.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                                            cax=cax, orientation="horizontal", ticks=[0, .5, 1])
            cax.tick_params(colors=MUTED, labelsize=9)
            cax.set_xlabel(label, color=FG, size=10, labelpad=3)
        f.text(.025, .012, "Fixed scales 0–1 · cell densities below 0.02 K hidden · one shared time and slice position",
               color=MUTED, size=10)
        f.text(.975, .012, "BrainWeb / MNI, McGill", color=MUTED, size=10, ha="right")
        for text in f.findobj(matplotlib.text.Text):
            text.set_fontsize(text.get_fontsize() * width / 1920)

    def _planes(self, view, i):
        _, histories, volumes, _ = self.data[view]
        return get_planes(volumes[i], self.indices) if self.custom_slices else {name: histories[name][i] for name in self.names}

    def frame(self, i):
        for view in self.fields:
            planes = self._planes(view, i)
            for name, image in self.images[view].items():
                image.set_data(overlay_rgba(planes[name], self.data[view][3], cmap=VIEW_STYLES[view][2],
                    threshold=0 if view == "oxygen" else .02,
                    mask=self.masks[name] if view == "oxygen" else None))
        m = self.r.metrics
        self.day.set_text(f"DAY {self.r.times[i]:05.1f}")
        self.stats["viable"].set_text(f"Viable cells: {m['cells'][i]:.3e}")
        self.stats["oxygen"].set_text(f"Minimum tissue oxygen: {m['min_oxygen'][i]:.3f}")
        self.stats["necrosis"].set_text(f"Necrotic equivalents: {m['necrotic_cells'][i]:.3e}")
        self.stats["reference"].set_text(f"Reference cells: {self.r.comparison.metrics['cells'][i]:.3e}")
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
    for field in ("oxygen", "necrosis"):
        planes = getattr(result, field + "_planes")
        if planes is not None:
            arrays.update({field + "_plane_" + key: value for key, value in planes.items()})
        final = getattr(result, "final_" + field)
        filename = output / ("final_oxygen_relative.nii.gz" if field == "oxygen" else "final_necrosis_cells_mm3.nii.gz")
        if final is not None:
            nib.save(nib.Nifti1Image(final, result.affine), filename)
        elif filename.exists():
            filename.unlink()
    if result.tissue_mask is not None:
        arrays["tissue_mask"] = result.tissue_mask
    np.savez_compressed(output / "simulation.npz", **arrays)
    nib.save(nib.Nifti1Image(result.final_density, result.affine), output / "final_density_cells_mm3.nii.gz")
    header = "time_days," + ",".join(result.metrics)
    np.savetxt(output / "growth_metrics.csv", np.column_stack([result.times, *result.metrics.values()]),
               delimiter=",", header=header, comments="")
    if result.comparison is not None:
        save_result(result.comparison, output / "reference_fisher_kpp")


def load_result(output):
    output = Path(output)
    with np.load(output / "simulation.npz", allow_pickle=False) as a:
        p, meta = json.loads(str(a["parameters"])), json.loads(str(a["metadata"]))
        result = Result(a["times"], {key[6:]: a[key] for key in a if key.startswith("plane_")},
                      {key[11:]: a[key] for key in a if key.startswith("background_")},
                      np.asarray(nib.load(output / "final_density_cells_mm3.nii.gz").dataobj),
                      {key[7:]: a[key] for key in a if key.startswith("metric_")},
                      p, meta, a["affine"], tuple(meta["indices"]))
        for field, filename in (("oxygen", "final_oxygen_relative.nii.gz"), ("necrosis", "final_necrosis_cells_mm3.nii.gz")):
            prefix = field + "_plane_"
            planes = {key[len(prefix):]: a[key] for key in a if key.startswith(prefix)}
            if planes:
                setattr(result, field + "_planes", planes)
                setattr(result, "final_" + field, np.asarray(nib.load(output / filename).dataobj))
        if "tissue_mask" in a:
            result.tissue_mask = a["tissue_mask"]
    if meta.get("comparison") == "reference_fisher_kpp":
        result.comparison = load_result(output / "reference_fisher_kpp")
    return result


def export_media(result, output, *, fps=12, width=1920, formats=("mp4", "gif"), progress=None, view="viable",
                 plane="all", indices=None, brain=None):
    if not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if set(formats) - {"mp4", "gif"}:
        raise ValueError("Formats must be mp4 and/or gif")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    renderer = (ComparisonRenderer(result, width, plane=plane, indices=indices, brain=brain)
                if view == "comparison" else SlideRenderer(result, width, view=view))
    stem = "tumor_growth" if view == "viable" else view
    poster = "poster.png" if view == "viable" else f"{view}_poster.png"
    poster_frame = renderer.frame(len(result.times) - 1)
    height = poster_frame.shape[0]
    Image.fromarray(poster_frame).save(output / poster)
    # H.264 + yuv420p + faststart, compatible with common slide software.
    writer = None
    if "mp4" in formats:
        writer = imageio.get_writer(output / f"{stem}.mp4", fps=fps, codec="libx264",
                                    pixelformat="yuv420p",
                                    quality=8, macro_block_size=1,
                                    output_params=["-movflags", "+faststart"])
    gif_frames = []
    # One shared palette across the animation prevents per-frame palette flicker.
    gif_width = min(width, 960)
    gif_height = round(gif_width * height / width)
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
        gif_frames[0].save(output / f"{stem}.gif", save_all=True, append_images=gif_frames[1:],
                           duration=round(1000 / fps / 10) * 10, loop=0, optimize=False, disposal=1)
    media = {"fps": fps, "width": width, "height": height,
             "gif_width": gif_width, "display_threshold_n_over_K": None if view == "oxygen" else .02,
             "simulation_days": float(result.times[-1]), "video_seconds": len(order) / fps,
             "formats": list(formats), "codec": "H.264", "pixel_format": "yuv420p"}
    media.update(view=view, color_scale=({field: VIEW_STYLES[field][1] for field in renderer.fields}
                                       if view == "comparison" else VIEW_STYLES[view][1]))
    if view == "comparison":
        media.update(fields=list(renderer.fields), planes=list(renderer.names),
                     slice_indices=list(renderer.indices), slice_xyz_mm=renderer.xyz.tolist(),
                     saved_times_days=result.times.tolist(), rendered_frame_indices=order,
                     rendered_times_days=[float(result.times[i]) for i in order], synchronized=True)
    if gif_frames:
        with Image.open(output / f"{stem}.gif") as gif:
            durations = []
            for i in range(gif.n_frames):
                gif.seek(i)
                durations.append(gif.info.get("duration", 0))
            media.update(gif_frames=gif.n_frames, gif_seconds=sum(durations) / 1000)
    metadata_name = "media_metadata.json" if view == "viable" else f"{view}_media_metadata.json"
    (output / metadata_name).write_text(json.dumps(media, indent=2) + "\n")
    return output
