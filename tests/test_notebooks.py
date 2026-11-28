"""Execute the example notebooks and fail if any cell raises an error."""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / 'examples'

# These notebooks are slow or need extra packages, so they are not run.
SKIPPED = {'Internal_stability_SO.ipynb',
           'Introduction to NodePy.ipynb',
           'stability_polynomial_speed.ipynb'}

NOTEBOOKS = sorted(p.name for p in EXAMPLES.glob('*.ipynb') if p.name not in SKIPPED)


@pytest.mark.notebook
@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook(name, monkeypatch):
    import nbclient
    import nbformat

    # Make the kernel import this checkout of nodepy rather than an installed
    # copy.  The kernel inherits the environment of this process.
    pythonpath = os.environ.get('PYTHONPATH')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(REPO)] + ([pythonpath] if pythonpath else [])))

    nb = nbformat.read(EXAMPLES / name, as_version=nbformat.current_nbformat)
    client = nbclient.NotebookClient(nb, timeout=120, kernel_name='python3', allow_errors=True,
                                     resources={'metadata': {'path': str(EXAMPLES)}})
    client.execute()

    errors = ['{}: {}'.format(output.ename, output.evalue)
              for cell in nb.cells for output in cell.get('outputs', [])
              if output.output_type == 'error']
    assert not errors, '\n'.join(errors)
