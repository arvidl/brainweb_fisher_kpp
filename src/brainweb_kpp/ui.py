"""ipywidgets/ipympl application. Import after `%matplotlib widget`."""
from pathlib import Path
import traceback
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display

from .data import load_brainweb, verify_data, ROOT
from .model import Parameters, fields, initial_condition, simulate, get_planes
from .render import display_plane, export_media, save_result


class TumorExplorer:
    def __init__(self):
        verify_data()
        self.brain = load_brainweb(2)
        self.result = None
        self.controls = {}
        self._busy = False
        p = Parameters()

        def slider(name, label, value, low, high, step):
            control = w.FloatSlider(description=label, value=value, min=low, max=high,
                                    step=step, continuous_update=False,
                                    style={"description_width": "150px"}, layout=w.Layout(width="100%", max_width="600px"))
            self.controls[name] = control
            return control

        coords = [slider("origin_" + axis, f"Origin {axis} (mm)", p.origin_mm[j], low, high, 1)
                  for j, (axis, low, high) in enumerate((("x", -90, 90), ("y", -126, 90), ("z", -72, 108)))]
        radii = [slider("radius_" + axis, f"Radius {axis} (mm)", 5, 1, 25, .5) for axis in "xyz"]
        self.controls["seed_profile"] = w.Dropdown(description="Seed profile", options=[("Gaussian (radii = σ)", "gaussian"),
                                                                                      ("Uniform ellipsoid", "ellipsoid")])
        self.controls["initial_density"] = w.BoundedFloatText(description="Peak cells/mm³", value=p.initial_density,
                                                               min=1, max=1e7, step=1000, style={"description_width": "150px"})
        seed = w.VBox([w.HTML("<b>RAS coordinates:</b> x right, y anterior, z superior. Click an MRI plane to place the seed.<br>"
                             "Use tissue preview to keep the center in gray/white matter. Gaussian is cut off at 3σ."),
                       w.GridBox([w.VBox(coords), w.VBox(radii)],
                                 layout=w.Layout(grid_template_columns="repeat(2, minmax(0, 1fr))", grid_gap="8px")),
                       self.controls["seed_profile"], self.controls["initial_density"]])
        transport = w.VBox([
            slider("d_white", "D white (mm²/day)", p.d_white, 0, 3, .05),
            slider("d_gray", "D gray (mm²/day)", p.d_gray, 0, 3, .05),
            slider("d_csf", "D CSF (mm²/day)", 0, 0, 1, .01),
            slider("spatial_multiplier", "D at origin ×", 1, .1, 4, .1),
            slider("spatial_width_mm", "D field σ (mm)", 30, 2, 100, 1),
            slider("rho", "ρ (1/day)", p.rho, 0, .3, .005),
            w.HTML("D(x) = D_tissue × [1 + (multiplier − 1) exp(−|x − origin|² / (2σ²))].<br>"
                   "CSF never proliferates. Class 8 (glial matter) uses gray matter coefficients."),
        ])
        self.controls["carrying_capacity"] = w.BoundedFloatText(description="K (cells/mm³)", value=p.carrying_capacity,
                                                                  min=1, max=1e7, step=10000, style={"description_width": "150px"})
        transport.children += (self.controls["carrying_capacity"],)
        self.dx = w.Dropdown(description="Δx (mm)", options=[.5, 1., 1.5, 2., 3., 4.], value=2.)
        self.controls["auto_substep"] = w.Checkbox(description="Automatically subdivide unstable time steps", value=True,
                                                   indent=False)
        self.store = w.Checkbox(description="Keep 3D frames for arbitrary slice playback (uses more RAM)", value=True, indent=False)
        numerics = w.VBox([self.dx,
                           slider("dt_days", "Max Δt (days)", p.dt_days, .01, 2, .01),
                           slider("duration_days", "Duration (days)", p.duration_days, 1, 365, 1),
                           slider("frame_days", "Save every (days)", 3, .5, 10, .5),
                           self.controls["auto_substep"], self.store,
                           w.HTML("The solver reports the actual step and divides it when necessary. 2 mm is a quick preview;<br>"
                                  "1 mm uses native anatomy. 0.5 mm interpolates the 1 mm phantom and costs substantially more.")])
        self.outpath = w.Text(description="Output folder", value="outputs/notebook_run", layout=w.Layout(width="650px"))
        self.fps = w.BoundedIntText(description="Frames/second", value=12, min=1, max=60)
        self.export_button = w.Button(description="Export MP4 + GIF", icon="film", disabled=True)
        export = w.VBox([self.outpath, self.fps, self.export_button,
                         w.HTML("Exports 1920 × 1080 H.264 MP4, 960 × 540 looping GIF, PNG poster, NIfTI density,<br>"
                                "replay data, parameters, metrics CSV and provenance. Existing files in this folder are replaced.")])
        tabs = w.Tab(children=[seed, transport, numerics, export])
        for i, title in enumerate(("1 · Origin & density", "2 · Diffusion & growth", "3 · Grid & time", "4 · Export")):
            tabs.set_title(i, title)
        self.run_button = w.Button(description="Run 3D simulation", button_style="primary", icon="play")
        self.preview_mode = w.Dropdown(description="Preview", options=["Seed on T1", "Tissue labels", "Diffusion D(x)"])
        self.progress = w.IntProgress(value=0, min=0, max=100, description="Progress")
        self.status = w.HTML()
        self.error = w.Output()
        self.play = w.Play(value=0, min=0, max=0, interval=120, disabled=True)
        self.time = w.IntSlider(description="Frame", value=0, min=0, max=0, continuous_update=False)
        self.link = w.jslink((self.play, "value"), (self.time, "value"))
        self.view = [w.IntSlider(description=f"View {axis} (mm)", min=low, max=high, value=int(p.origin_mm[j]),
                                 continuous_update=False, layout=w.Layout(width="32%"))
                     for j, (axis, low, high) in enumerate((("x", -90, 90), ("y", -126, 90), ("z", -72, 108)))]
        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(8, 3.5), layout="constrained")
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_position = "bottom"
        self.fig.canvas.layout = w.Layout(width="100%", height="370px")
        self.colorbar = self.fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
                                         ax=self.axes, shrink=.65, pad=.015, label="n/K")
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.widget = w.VBox([w.HTML("<h2>BrainWeb · Fisher–KPP growth explorer</h2>"), tabs,
                              w.HBox([self.run_button, self.preview_mode, self.progress]), self.status,
                              self.fig.canvas, w.HBox([self.play, self.time]),
                              w.HBox(self.view), self.error], layout=w.Layout(width="100%"))
        for control in [*self.controls.values(), self.dx, self.store]:
            control.observe(self._changed, names="value")
        for control in [self.preview_mode, self.time, *self.view]:
            control.observe(self._redraw, names="value")
        self.run_button.on_click(self._run)
        self.export_button.on_click(self._export)
        self._redraw()

    def parameters(self):
        values = {name: c.value for name, c in self.controls.items() if not name.startswith(("origin_", "radius_"))}
        values["origin_mm"] = tuple(self.controls["origin_" + axis].value for axis in "xyz")
        values["radii_mm"] = tuple(self.controls["radius_" + axis].value for axis in "xyz")
        return Parameters(**values)

    def _changed(self, event=None):
        if self._busy: return
        self.result = None
        self.export_button.disabled = True
        self.play.disabled = True
        self.time.max = 0
        self.status.value = "Parameters changed. Press <b>Run 3D simulation</b> to apply."
        self._redraw()

    def _click(self, event):
        if self._busy or event.inaxes not in self.axes or event.xdata is None or self.result is not None: return
        index = list(self.axes).index(event.inaxes)
        pairs = (("x", "y"), ("x", "z"), ("y", "z"))
        self._busy = True
        for axis, coord in zip(pairs[index], (event.xdata, event.ydata)):
            c = self.controls["origin_" + axis]
            c.value = np.clip(round(coord), c.min, c.max)
        self._busy = False
        self._changed()

    def _redraw(self, event=None):
        if self._busy: return
        p = self.parameters()
        try:
            if self.result is not None:
                i = self.time.value
                if self.result.volumes is not None:
                    ijk = self.brain.voxel(tuple(c.value for c in self.view))
                    planes = get_planes(self.result.volumes[i], ijk)
                    backgrounds = get_planes(self.brain.t1, ijk)
                else:
                    ijk = self.result.indices
                    planes = {k: v[i] for k, v in self.result.planes.items()}
                    backgrounds = self.result.backgrounds
                title = f"Day {self.result.times[i]:g}  ·  Density n/K (values below 0.02 hidden)"
            else:
                ijk = self.brain.voxel(p.origin_mm)
                backgrounds = get_planes(self.brain.t1, ijk)
                try:
                    planes = get_planes(initial_condition(self.brain, p), ijk)
                    title = "Initial tumor density n/K  ·  fixed inferno color scale 0–1"
                except ValueError as e:
                    planes = get_planes(np.zeros_like(self.brain.t1), ijk)
                    title = str(e)
            coords = self.brain.world(ijk)
            orientations = (("L", "R", "P", "A"), ("L", "R", "I", "S"), ("P", "A", "I", "S"))
            dims = ((0, 1), (0, 2), (1, 2))
            for j, (ax, name) in enumerate(zip(self.axes, ("Axial", "Coronal", "Sagittal"))):
                ax.clear()
                mode = self.preview_mode.value if self.result is None else "Seed on T1"
                if mode == "Seed on T1":
                    display_plane(ax, backgrounds[name], planes[name], p.carrying_capacity,
                                  spacing=self.brain.spacing, labels=orientations[j])
                else:
                    volume = self.brain.labels if mode == "Tissue labels" else fields(self.brain, p)[0]
                    plane = get_planes(volume, ijk)[name]
                    if mode == "Tissue labels":
                        cmap = ListedColormap(["black", "#4b9bd3", "#8d8d8d", "#f0f0f0", "#63413e", "#63413e", "#63413e", "#63413e", "#b5a1c9", "#63413e"])
                        ax.imshow(plane.T, origin="lower", cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 10), 10))
                        title = "Tissue: CSF blue · GM gray · WM white · glial purple · extracerebral brown"
                    else:
                        ax.imshow(plane.T, origin="lower", cmap="viridis", vmin=0,
                                  vmax=max(p.d_white, p.d_gray, p.d_csf) * max(1, p.spatial_multiplier) or 1)
                        title = "D(x) in mm²/day · viridis scale 0 to " + str(round(max(p.d_white, p.d_gray, p.d_csf) * max(1, p.spatial_multiplier), 3))
                    ax.set_axis_off()
                a, b = dims[j]
                extent = (self.brain.affine[a, 3] - self.brain.spacing / 2,
                          self.brain.affine[a, 3] + (backgrounds[name].shape[0] - .5) * self.brain.spacing,
                          self.brain.affine[b, 3] - self.brain.spacing / 2,
                          self.brain.affine[b, 3] + (backgrounds[name].shape[1] - .5) * self.brain.spacing)
                for image in ax.images: image.set_extent(extent)
                ax.set_title(f"{name} · {'zyx'[j]} = {coords[(2, 1, 0)[j]]:g} mm")
            mode = self.preview_mode.value if self.result is None else "Seed on T1"
            self.colorbar.ax.set_visible(mode != "Tissue labels")
            vmax = max(p.d_white, p.d_gray, p.d_csf) * max(1, p.spatial_multiplier) or 1
            self.colorbar.update_normal(ScalarMappable(norm=Normalize(0, vmax if mode == "Diffusion D(x)" else 1),
                                                      cmap="viridis" if mode == "Diffusion D(x)" else "inferno"))
            self.colorbar.set_label("D (mm²/day)" if mode == "Diffusion D(x)" else "n/K")
            self.fig.suptitle(title, fontsize=10)
            self.fig.canvas.draw_idle()
        except ValueError as e:
            self.status.value = str(e)

    def _run(self, button=None):
        self._busy = True
        self.run_button.disabled = self.export_button.disabled = True
        self.error.clear_output()
        self.result = None
        try:
            self.status.value = "Resampling anatomy and solving in 3D… First run also compiles the solver."
            p = self.parameters()
            self.brain = load_brainweb(self.dx.value)
            def progress(done, total): self.progress.value = round(100 * done / total)
            self.result = simulate(self.brain, p, store_volumes=self.store.value, progress=progress)
            self.time.max = self.play.max = len(self.result.times) - 1
            self.time.value = len(self.result.times) - 1
            self.play.disabled = False
            for j, c in enumerate(self.view): c.value = round(p.origin_mm[j])
            self.export_button.disabled = False
            m = self.result.metadata
            self.status.value = (f"<b>Completed:</b> {m['internal_steps']} internal steps · actual maximum Δt "
                                 f"{m['actual_max_dt_days']:.4g} days · {m['elapsed_seconds']:.1f} seconds · "
                                 f"{len(self.result.times)} frames. "
                                 + ("View sliders explore saved 3D volumes." if self.store.value else "Playback uses planes through the origin."))
        except Exception as e:
            self.status.value = f"<b>Could not run:</b> {e}"
            with self.error: traceback.print_exc()
        finally:
            self.run_button.disabled = False
            self._busy = False
            self._redraw()

    def _export(self, button=None):
        if self.result is None: return
        self.export_button.disabled = self.run_button.disabled = True
        try:
            output = Path(self.outpath.value).expanduser()
            if not output.is_absolute(): output = ROOT / output
            self.status.value = "Saving simulation and rendering MP4 / GIF…"
            save_result(self.result, output)
            export_media(self.result, output, fps=self.fps.value,
                         progress=lambda done, total: setattr(self.progress, "value", round(100 * done / total)))
            self.status.value = f"<b>Saved MP4, GIF and reproducibility files:</b> {output}"
        except Exception as e:
            self.status.value = f"<b>Export failed:</b> {e}"
            with self.error: traceback.print_exc()
        finally:
            self.export_button.disabled = self.run_button.disabled = False
