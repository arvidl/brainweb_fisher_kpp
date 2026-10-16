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
print('Widget callbacks, simulation and slice playback passed.')
"""))
NotebookClient(notebook, timeout=180, kernel_name="python3",
               resources={"metadata": {"path": str(root)}}).execute()
print("Notebook execution and short widget simulation passed.")
