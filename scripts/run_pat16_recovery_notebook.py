"""Execute the PAT16 recovery notebook, replaying records or refitting with --rerun."""
from pathlib import Path
import argparse
import os

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun', action='store_true', help='Refit all bootstrap/noise trials using real PAT16 inputs')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/brainweb-mpl')
    os.environ.setdefault('IPYTHONDIR', '/tmp/brainweb-ipython')
    root = Path(__file__).resolve().parents[1]
    path = root/'pat16_synthetic_parameter_recovery.ipynb'
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'RERUN = ' in cell.source:
            lines = cell.source.splitlines()
            cell.source = '\n'.join(f'RERUN = {args.rerun}  # True refits; False replays checked records'
                                    if line.startswith('RERUN = ') else line for line in lines)

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    print(output.text, end='', flush=True)
            print(f'Finished notebook cell {cell_index}', flush=True)

    NotebookClient(notebook, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}},
                   on_cell_executed=report_cell).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print('Saved executed PAT16 recovery notebook.', flush=True)


if __name__ == '__main__':
    main()
