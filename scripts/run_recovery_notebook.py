"""Execute and save the synthetic inference notebook with the project kernel.

Run from the repository root with:
    uv run --locked python scripts/run_recovery_notebook.py
Requires permission to open local Jupyter kernel sockets.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
path = root / "synthetic_parameter_recovery.ipynb"
notebook = nbformat.read(path, as_version=4)


def report_cell(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="", flush=True)


NotebookClient(notebook, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(root)}},
               on_cell_executed=report_cell).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print("Executed notebook saved; results in outputs/synthetic_recovery/.")
