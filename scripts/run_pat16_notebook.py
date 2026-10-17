"""Execute the PAT16 pilot using the current project Python kernel."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brainweb-mpl')
root = Path(__file__).resolve().parents[1]
path = root / 'pat16_anisotropic_pilot.ipynb'
notebook = nbformat.read(path, as_version=4)
def report_cell(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='', flush=True)
        print(f'Finished code cell at notebook index {cell_index}', flush=True)
NotebookClient(notebook, timeout=300, kernel_name='python3',
               resources={'metadata': {'path': str(root)}},
               on_cell_executed=report_cell).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print('Executed PAT16 notebook saved.')
