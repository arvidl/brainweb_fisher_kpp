"""ipywidgets/ipympl application. Import after `%matplotlib widget`."""
from pathlib import Path
from io import BytesIO
import traceback
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display

from .data import load_brainweb, verify_data, ROOT
from .model import Parameters, fields, initial_condition, simulate, get_planes
from .render import display_plane, export_media, save_result, field_data, VIEW_STYLES


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
        self.controls["model"] = w.Dropdown(description="Model", value=p.model,
            options=[("Original Fisher–KPP", "fisher_kpp"), ("Tumor + oxygen + necrosis", "coupled")],
            style={"description_width": "150px"}, layout=w.Layout(width="520px"))
        self.compare = w.Checkbox(description="Also run Fisher–KPP reference with the same seed and transport", value=True, indent=False)
        oxygen_controls = [
            slider("oxygen_length_mm", "O₂ length (mm)", p.oxygen_length_mm, 0, 3, .05),
            slider("oxygen_uptake_ratio", "O₂ uptake β", p.oxygen_uptake_ratio, 0, 20, .25),
            slider("perfusion_suppression", "Perfusion loss α", p.perfusion_suppression, 0, 20, .25),
            slider("oxygen_prolif_level", "O₂ full growth θp", p.oxygen_prolif_level, .02, .99, .01),
            slider("oxygen_death_level", "O₂ death θd", p.oxygen_death_level, .01, .98, .01),
            slider("necrosis_rate", "Death δ (1/day)", p.necrosis_rate, 0, 2, .05),
            slider("necrosis_clearance", "Clearance (1/day)", p.necrosis_clearance, 0, .2, .005),
        ]
        model_tab = w.VBox([self.controls["model"], self.compare, *oxygen_controls,
            w.HTML("Oxygen is solved at quasi-steady state. Necrotic material occupies growth capacity.<br>"
                   "Require 0 &lt; θd &lt; θp &lt; 1. Coupled transport excludes CSF; its D control is ignored.<br>"
                   "Oxygen is relative to normal tissue (1); new defaults are illustrative, not calibrated.<br>"
                   "The reference changes the reaction model while keeping seed, grid and transport identical.")])
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
                         w.HTML("Exports the current display as MP4, GIF and poster. Side-by-side exports keep all four fields<br>"
                                "at the same day and displayed slices; single-field exports use the seed planes.<br>"
                                "All available fields are saved "
                                "as NIfTI and replay planes, with parameters, metrics and reference results.<br>"
                                "Existing files in this folder are replaced.")])
        tabs = w.Tab(children=[seed, transport, model_tab, numerics, export])
        for i, title in enumerate(("1 · Origin & density", "2 · Diffusion & growth", "3 · Model & oxygen", "4 · Grid & time", "5 · Export")):
            tabs.set_title(i, title)
        self.tabs = tabs
        self.settings = w.Accordion(children=[tabs], titles=("Simulation parameters & export",), selected_index=0)
        self.run_button = w.Button(description="Run 3D simulation", button_style="primary", icon="play")
        self.preview_mode = w.Dropdown(description="Preview", options=["Seed on T1", "Tissue labels", "Diffusion D(x)"])
        self.result_view = w.Dropdown(description="Result view", options=[("Viable tumor", "viable")], disabled=True,
                                     layout=w.Layout(width="340px"))
        self.display_layout = w.Dropdown(description="Display", value="single", disabled=True,
            options=[("Single field", "single"), ("Side by side", "comparison")],
            layout=w.Layout(width="260px"))
        self.comparison_plane = w.Dropdown(description="Planes", value="all",
            options=[("All three planes", "all"), ("Axial", "Axial"),
                     ("Coronal", "Coronal"), ("Sagittal", "Sagittal")],
            layout=w.Layout(width="260px", display="none"))
        self.summary = w.HTML()
        self.progress = w.IntProgress(value=0, min=0, max=100, description="Progress")
        self.status = w.HTML()
        self.error = w.Output()
        self.play = w.Play(value=0, min=0, max=0, interval=120, disabled=True)
        self.time = w.IntSlider(description="Frame", value=0, min=0, max=0, continuous_update=False,
                               layout=w.Layout(flex="1 1 220px", min_width="220px"))
        self.day = w.HTML()
        self.link = w.jslink((self.play, "value"), (self.time, "value"))
        self.view = [w.IntSlider(description=f"View {axis} (mm)", min=low, max=high, value=int(p.origin_mm[j]),
                                 continuous_update=False, style={"description_width": "90px"},
                                 layout=w.Layout(flex="1 1 220px", min_width="220px"))
                     for j, (axis, low, high) in enumerate((("x", -90, 90), ("y", -126, 90), ("z", -72, 108)))]
        with plt.ioff():
            self.fig = plt.figure(figsize=(8, 3.5), layout="constrained")
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_position = "bottom"
        self._figure_layout = None
        self.comparison_axes = {}
        self.comparison_colorbars = {}
        self._configure_figure("single")
        # A responsive PNG lets all columns fit narrow notebook/browser panels.
        # The shared live widgets still redraw it from the saved 3D fields.
        self.comparison_image = w.Image(format="png", layout=w.Layout(
            width="100%", max_width="1450px", height="auto", object_fit="contain", display="none"))
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        wrap = lambda: w.Layout(display="flex", flex_flow="row wrap", align_items="center")
        self.widget = w.VBox([w.HTML("<h2>BrainWeb · Tumor growth explorer</h2>"), self.settings,
                              w.HBox([self.run_button, self.preview_mode, self.progress], layout=wrap()), self.status,
                              w.HBox([self.display_layout, self.result_view, self.comparison_plane], layout=wrap()),
                              w.HBox([self.play, self.time, self.day], layout=wrap()),
                              w.HBox(self.view, layout=wrap()), self.summary,
                              self.fig.canvas, self.comparison_image, self.error], layout=w.Layout(width="100%", min_width="0"))
        for control in [*self.controls.values(), self.dx, self.store, self.compare]:
            control.observe(self._changed, names="value")
        for control in [self.preview_mode, self.result_view, self.display_layout, self.comparison_plane, self.time, *self.view]:
            control.observe(self._redraw, names="value")
        self.run_button.on_click(self._run)
        self.export_button.on_click(self._export)
        self._model_controls()
        self._redraw()

    def parameters(self):
        values = {name: c.value for name, c in self.controls.items() if not name.startswith(("origin_", "radius_"))}
        values["origin_mm"] = tuple(self.controls["origin_" + axis].value for axis in "xyz")
        values["radii_mm"] = tuple(self.controls["radius_" + axis].value for axis in "xyz")
        return Parameters(**values)

    def _changed(self, event=None):
        if self._busy: return
        self._model_controls()
        self.result = None
        self.export_button.disabled = True
        self.play.disabled = True
        self.time.max = 0
        self.play.max = 0
        self.result_view.disabled = True
        self.display_layout.disabled = True
        self.display_layout.value = "single"
        self.preview_mode.disabled = False
        self.summary.value = ""
        self.day.value = ""
        self.status.value = "Parameters changed. Press <b>Run 3D simulation</b> to apply."
        self._redraw()

    def _model_controls(self):
        coupled = self.controls["model"].value == "coupled"
        for name in ("oxygen_length_mm", "oxygen_uptake_ratio", "perfusion_suppression",
                     "oxygen_prolif_level", "oxygen_death_level", "necrosis_rate", "necrosis_clearance"):
            self.controls[name].disabled = not coupled
        self.compare.disabled = not coupled
        self.controls["d_csf"].disabled = coupled

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

    def _configure_figure(self, mode, views=(), planes=()):
        """Rebuild axes only when the display layout changes, never per frame."""
        key = (mode, tuple(views), tuple(planes))
        if self._figure_layout == key:
            return
        self.fig.clear()
        self.comparison_axes = {}
        self.comparison_colorbars = {}
        if mode == "single":
            self.fig.set_size_inches(8, 3.5)
            self.axes = list(self.fig.subplots(1, 3))
            self.colorbar = self.fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
                                             ax=self.axes, shrink=.65, pad=.015, label="c / K")
            height = "370px"
        else:
            rows = len(planes)
            self.fig.set_size_inches(3.1 * len(views), 2.65 * rows + 1.1)
            axes = self.fig.subplots(rows, len(views), squeeze=False)
            self.axes = list(axes.flat)
            for col, view in enumerate(views):
                self.comparison_axes[view] = dict(zip(planes, axes[:, col]))
                _, label, cmap = VIEW_STYLES[view]
                colorbar = self.fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                    ax=list(axes[:, col]), orientation="horizontal", shrink=.85, pad=.025,
                    fraction=.035, ticks=[0, .5, 1], label=label)
                colorbar.ax.tick_params(labelsize=8)
                colorbar.ax.xaxis.label.set_size(8)
                self.comparison_colorbars[view] = colorbar
            height = f"{285 * rows + 100}px"
        self.fig.canvas.layout = w.Layout(width="100%", min_width="0", height=height)
        self._figure_layout = key

    def _result_planes(self, view):
        """Every field uses this same frame index and RAS slice position."""
        _, histories, volumes, scale = field_data(self.result, view)
        if volumes is not None:
            ijk = self.brain.voxel(tuple(c.value for c in self.view))
            planes = get_planes(volumes[self.time.value], ijk)
            backgrounds = get_planes(self.brain.t1, ijk)
        else:
            ijk = self.result.indices
            planes = {k: v[self.time.value] for k, v in histories.items()}
            backgrounds = self.result.backgrounds
        masks = get_planes(self.result.tissue_mask, ijk) if view == "oxygen" else None
        return planes, backgrounds, masks, scale, ijk

    def _world_extent(self, name, shape):
        a, b = {"Axial": (0, 1), "Coronal": (0, 2), "Sagittal": (1, 2)}[name]
        return (self.brain.affine[a, 3] - self.brain.spacing / 2,
                self.brain.affine[a, 3] + (shape[0] - .5) * self.brain.spacing,
                self.brain.affine[b, 3] - self.brain.spacing / 2,
                self.brain.affine[b, 3] + (shape[1] - .5) * self.brain.spacing)

    def _redraw_comparison(self):
        views = tuple(value for _, value in self.result_view.options)
        names = ("Axial", "Coronal", "Sagittal") if self.comparison_plane.value == "all" else (self.comparison_plane.value,)
        self._configure_figure("comparison", views, names)
        orientations = {"Axial": ("L", "R", "P", "A"), "Coronal": ("L", "R", "I", "S"),
                        "Sagittal": ("P", "A", "I", "S")}
        for view in views:
            planes, backgrounds, masks, scale, ijk = self._result_planes(view)
            coords = self.brain.world(ijk)
            for row, name in enumerate(names):
                ax = self.comparison_axes[view][name]
                ax.clear()
                display_plane(ax, backgrounds[name], planes[name], scale, spacing=self.brain.spacing,
                    labels=orientations[name], cmap=VIEW_STYLES[view][2],
                    threshold=0 if view == "oxygen" else .02,
                    mask=masks[name] if masks is not None else None)
                extent = self._world_extent(name, backgrounds[name].shape)
                for image in ax.images:
                    image.set_extent(extent)
                axis, coord = {"Axial": ("z", coords[2]), "Coronal": ("y", coords[1]),
                               "Sagittal": ("x", coords[0])}[name]
                title = f"{name} · {axis} = {coord:g} mm"
                if row == 0:
                    title = VIEW_STYLES[view][0] + "\n" + title
                ax.set_title(title, fontsize=9)
        self.fig.suptitle(f"Day {self.result.times[self.time.value]:g} · synchronized time & slices\n"
                         "Fixed scales 0–1 · cell densities below 0.02 K hidden", fontsize=10)
        self.fig.canvas.layout.display = "none"
        self.comparison_image.layout.display = ""
        buffer = BytesIO()
        self.fig.savefig(buffer, format="png", dpi=110)
        self.comparison_image.value = buffer.getvalue()

    def _redraw(self, event=None):
        if self._busy: return
        p = self.parameters()
        try:
            comparison = self.result is not None and self.display_layout.value == "comparison"
            self.comparison_plane.layout.display = "" if comparison else "none"
            self.result_view.layout.display = "none" if comparison else ""
            if self.result is not None:
                i = self.time.value
                selected = self.result_view.value
                field_title, field_label, field_cmap = VIEW_STYLES[selected]
                planes, backgrounds, masks, scale, ijk = self._result_planes(selected)
                title = f"Day {self.result.times[i]:g} · {field_title} · {field_label}"
                if selected != "oxygen": title += " (below 0.02 hidden)"
                m = self.result.metrics
                model_label = "Coupled" if self.result.final_oxygen is not None else "Fisher–KPP"
                summary = f"{model_label} viable cells: {m['cells'][i]:.3e} · viable ≥ 0.2 K: {m['volume_20_ml'][i]:.2f} mL"
                if self.result.final_oxygen is not None:
                    summary += (f" · necrotic equivalents: {m['necrotic_cells'][i]:.3e}"
                                f" · minimum oxygen: {m['min_oxygen'][i]:.3f}"
                                f" · O₂ &lt; θd: {m['hypoxic_volume_ml'][i]:.2f} mL")
                if self.result.comparison is not None:
                    summary += f"<br>Fisher–KPP reference: {self.result.comparison.metrics['cells'][i]:.3e} viable cells · same day and slice"
                self.summary.value = summary
                self.day.value = f"<b>Day {self.result.times[i]:g} / {self.result.times[-1]:g}</b>"
                if comparison:
                    self._redraw_comparison()
                    return
            else:
                selected, scale, field_cmap, field_label, masks = "viable", p.carrying_capacity, "inferno", "c/K", None
                ijk = self.brain.voxel(p.origin_mm)
                backgrounds = get_planes(self.brain.t1, ijk)
                try:
                    planes = get_planes(initial_condition(self.brain, p), ijk)
                    title = "Initial tumor density n/K  ·  fixed inferno color scale 0–1"
                except ValueError as e:
                    planes = get_planes(np.zeros_like(self.brain.t1), ijk)
                    title = str(e)
            self._configure_figure("single")
            self.fig.canvas.layout.display = ""
            self.comparison_image.layout.display = "none"
            coords = self.brain.world(ijk)
            orientations = (("L", "R", "P", "A"), ("L", "R", "I", "S"), ("P", "A", "I", "S"))
            for j, (ax, name) in enumerate(zip(self.axes, ("Axial", "Coronal", "Sagittal"))):
                ax.clear()
                mode = self.preview_mode.value if self.result is None else "Seed on T1"
                if mode == "Seed on T1":
                    display_plane(ax, backgrounds[name], planes[name], scale,
                                  spacing=self.brain.spacing, labels=orientations[j], cmap=field_cmap,
                                  threshold=0 if selected == "oxygen" else .02,
                                  mask=masks[name] if masks is not None else None)
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
                extent = self._world_extent(name, backgrounds[name].shape)
                for image in ax.images: image.set_extent(extent)
                ax.set_title(f"{name} · {'zyx'[j]} = {coords[(2, 1, 0)[j]]:g} mm")
            mode = self.preview_mode.value if self.result is None else "Seed on T1"
            self.colorbar.ax.set_visible(mode != "Tissue labels")
            vmax = max(p.d_white, p.d_gray, p.d_csf) * max(1, p.spatial_multiplier) or 1
            self.colorbar.update_normal(ScalarMappable(norm=Normalize(0, vmax if mode == "Diffusion D(x)" else 1),
                                                      cmap="viridis" if mode == "Diffusion D(x)" else field_cmap))
            self.colorbar.set_label("D (mm²/day)" if mode == "Diffusion D(x)" else field_label)
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
            self.result = simulate(self.brain, p, store_volumes=self.store.value, progress=progress,
                                   compare=self.compare.value and p.model == "coupled")
            options = [("Viable tumor", "viable")]
            if self.result.final_oxygen is not None:
                options.extend([("Oxygen availability", "oxygen"), ("Necrotic material", "necrosis")])
            if self.result.comparison is not None: options.append(("Fisher–KPP reference", "reference"))
            self.result_view.options = options
            self.result_view.value = "viable"
            self.result_view.disabled = False
            self.display_layout.disabled = self.result.final_oxygen is None
            self.preview_mode.disabled = True
            self.time.max = self.play.max = len(self.result.times) - 1
            self.time.value = len(self.result.times) - 1
            self.play.disabled = False
            for j, c in enumerate(self.view):
                c.value = round(p.origin_mm[j])
                c.disabled = not self.store.value
            self.export_button.disabled = False
            m = self.result.metadata
            self.status.value = (f"<b>Completed:</b> {m['internal_steps']} internal steps · actual maximum Δt "
                                 f"{m['actual_max_dt_days']:.4g} days · {m.get('elapsed_seconds_with_reference', m['elapsed_seconds']):.1f} seconds · "
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
            comparison = self.display_layout.value == "comparison"
            options = (dict(view="comparison", plane=self.comparison_plane.value,
                            indices=self.brain.voxel(tuple(c.value for c in self.view)), brain=self.brain)
                       if comparison else dict(view=self.result_view.value))
            export_media(self.result, output, fps=self.fps.value,
                         progress=lambda done, total: setattr(self.progress, "value", round(100 * done / total)),
                         **options)
            self.status.value = f"<b>Saved MP4, GIF and reproducibility files:</b> {output}"
        except Exception as e:
            self.status.value = f"<b>Export failed:</b> {e}"
            with self.error: traceback.print_exc()
        finally:
            self.export_button.disabled = self.run_button.disabled = False
