"""Execute notebook + a short widget-callback run (requires local kernel sockets)."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "tumor_growth_explorer.ipynb", as_version=4)
notebook.cells.append(nbformat.v4.new_code_cell("""
app.controls['duration_days'].value = 6
app.controls['frame_days'].value = 2
app._run()
assert app.result is not None, app.status.value
assert app.result.times[-1] == 6
assert not app.export_button.disabled
app.time.value = 0
app.view[0].value = -28
assert len(app.axes[0].images) == 2
app.controls['model'].value = 'coupled'
assert not app.controls['oxygen_uptake_ratio'].disabled
assert app.controls['d_csf'].disabled
app.controls['initial_density'].value = 80000
app.controls['duration_days'].value = 2
app.controls['frame_days'].value = 1
app._run()
assert app.result is not None, app.status.value
assert app.result.final_oxygen is not None
assert app.result.comparison is not None
assert app.result.metrics['necrotic_cells'][-1] > 0
for view in ('viable', 'oxygen', 'necrosis', 'reference'):
    app.result_view.value = view
    app.time.value = 0
    app.view[0].value = -26
    app.time.value = 2
    assert len(app.axes[0].images) == 2
    assert view != 'oxygen' or app.colorbar.ax.get_ylabel() == 'o / normal baseline'
assert 'Fisher–KPP reference' in app.summary.value
import numpy as np
from brainweb_kpp.model import get_planes
from brainweb_kpp.render import field_data, overlay_rgba, VIEW_STYLES

def check_comparison():
    # Check actual displayed pixels against each saved field at the shared frame/slice.
    for view, axes in app.comparison_axes.items():
        _, histories, volumes, scale = field_data(app.result, view)
        ijk = app.brain.voxel(tuple(c.value for c in app.view)) if volumes is not None else app.result.indices
        planes = get_planes(volumes[app.time.value], ijk) if volumes is not None else {k: v[app.time.value] for k, v in histories.items()}
        mask = get_planes(app.result.tissue_mask, ijk) if view == 'oxygen' else None
        for name, ax in axes.items():
            expected = overlay_rgba(planes[name], scale, cmap=VIEW_STYLES[view][2],
                                    threshold=0 if view == 'oxygen' else .02,
                                    mask=mask[name] if mask is not None else None)
            np.testing.assert_allclose(ax.images[1].get_array(), expected)
            assert ax.images[0].get_extent() == ax.images[1].get_extent()
    assert app.comparison_colorbars['viable'].mappable.get_clim() == (0, 1)
    assert app.comparison_colorbars['reference'].mappable.get_clim() == (0, 1)

app.display_layout.value = 'comparison'
assert set(app.comparison_axes) == {'viable', 'oxygen', 'necrosis', 'reference'}
assert len(app.fig.axes) == 16  # 12 images + 4 fixed color scales
for frame, xyz in ((0, (-30, 25, 30)), (2, (-26, 23, 28))):
    for slider, coordinate in zip(app.view, xyz):
        slider.value = coordinate
    app.time.value = frame
    check_comparison()
    assert f'Day {app.result.times[frame]:g} /' in app.day.value
for plane in ('Axial', 'Coronal', 'Sagittal', 'all'):
    app.comparison_plane.value = plane
    check_comparison()
    assert len(app.fig.axes) == (16 if plane == 'all' else 8)
app.display_layout.value = 'single'
assert len(app.fig.axes) == 4  # axes/colorbars do not accumulate when switching
app.store.value = False
app._run()
assert app.result is not None, app.status.value
assert app.result.oxygen_volumes is None
for view in ('viable', 'oxygen', 'necrosis', 'reference'):
    app.result_view.value = view
    assert len(app.axes[0].images) == 2
assert all(slider.disabled for slider in app.view)
app.display_layout.value = 'comparison'
app.time.value = 0
check_comparison()
app.time.value = app.time.max
check_comparison()
app.controls['model'].value = 'fisher_kpp'
assert app.controls['oxygen_uptake_ratio'].disabled
assert not app.controls['d_csf'].disabled
assert app.result is None and app.export_button.disabled
assert app.display_layout.disabled and app.display_layout.value == 'single'
assert len(app.fig.axes) == 4
print('Both models, all field views, synchronized comparisons, full/slice-only playback and invalidation passed.')
"""))
NotebookClient(notebook, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(root)}}).execute()
print("Notebook execution and short widget simulation passed.")
