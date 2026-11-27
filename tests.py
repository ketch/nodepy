#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import doctest
import nodepy
import unittest
import os
import sys
import nbformat
from nbclient import NotebookClient

if sys.version_info >= (3,0):
    kernel = 'python3'
else:
    kernel = 'python2'

def _notebook_run(path):
    """Execute a notebook with nbclient and collect output.
       :returns (parsed nb object, execution errors)
    """
    # Make the notebook kernels import this checkout of nodepy rather than
    # an installed copy.  Kernels inherit the environment of this process.
    repo = os.path.dirname(os.path.abspath(__file__))
    old_pythonpath = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = os.pathsep.join([repo] + ([old_pythonpath] if old_pythonpath else []))
    try:
        nb = nbformat.read(path, as_version=nbformat.current_nbformat)
        client = NotebookClient(nb, timeout=120, kernel_name=kernel, allow_errors=True,
                                resources={'metadata': {'path': os.path.dirname(os.path.abspath(path))}})
        client.execute()
    finally:
        if old_pythonpath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = old_pythonpath

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors

def run_tests():
    retcode = 0
    for filename in os.listdir('./examples'):
        if (filename.split('.')[-1] == 'ipynb' and
            filename not in ['Internal_stability_SO.ipynb',
                             'Introduction to NodePy.ipynb',
                             'stability_polynomial_speed.ipynb']):
            print('running notebook: '+ filename)
            _, errors = _notebook_run('./examples/'+filename)
            if errors != []:
                retcode = 1
                for error in errors:
                    print('  {}: {}'.format(error.ename, error.evalue))

    for module_name in ['runge_kutta_method',
                        'linear_multistep_method',
                        'twostep_runge_kutta_method',
                        'downwind_runge_kutta_method',
                        'ivp',
                        'low_storage_rk',
                        'rooted_trees',
                        'snp',
                        'stability_function',
                        'general_linear_method',
                        'ode_solver',
                        'semidisc',
                        'strmanip',
                        'utils',
                        'graph',
                        'convergence',
                        'loadmethod']:
        module = nodepy.__getattribute__(module_name)
        result = doctest.testmod(module)
        if result.failed != 0:
            retcode = 1

    test_results = unittest.main(module='nodepy.unit_tests', exit=False)
    if not test_results.result.wasSuccessful():
        retcode = 1
    return retcode

if __name__ == '__main__':
    retcode = run_tests()
    sys.exit(retcode)
