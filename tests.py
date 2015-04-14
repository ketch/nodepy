#!/usr/bin/env python

import matplotlib
import doctest
import nodepy
matplotlib.use('agg')
import unittest

for module_name in ('runge_kutta_method',
                    'linear_multistep_method',
                    'twostep_runge_kutta_method',
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
                    'convergence'):
    module = nodepy.__getattribute__(module_name)
    doctest.testmod(module)

unittest.main(module='nodepy.unit_tests',exit=False)
