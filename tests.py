#!/usr/bin/env python

import matplotlib
import doctest
import nodepy
matplotlib.use('agg')
import unittest
import os
import subprocess
import tempfile
import sys
import nbformat

if sys.version_info >= (3,0):
    kernel = 'python3'
else:
    kernel = 'python2'

def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=60",
                "--ExecutePreprocessor.kernel_name="+kernel,
                "--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.reads(fout.read().decode('utf-8'), nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors

for filename in os.listdir('./examples'):
    if (filename.split('.')[-1] == 'ipynb' and
        filename not in ['Internal_stability_SO.ipynb','Introduction to NodePy.ipynb']):
        print('running ', filename)
        _, errors = _notebook_run('./examples/'+filename)
        if errors != []:
            raise(Exception)

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
    doctest.testmod(module)

unittest.main(module='nodepy.unit_tests',exit=False)
