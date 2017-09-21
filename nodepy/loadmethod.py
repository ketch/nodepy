r"""
Load some high-order RK pairs.

**Examples**::

    >>> from nodepy import loadmethod
    >>> myrk = loadmethod.load_rk_from_file()
    >>> myrk.order()
    12
    >>> myrk.name
    "Feagin's 12(10) pair"

"""
from __future__ import absolute_import
import os
import numpy as np

from nodepy import rk

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path,'method_coefficients')

def method_name_and_stages(file_name):
    if ((file_name=='rk108.txt')):
        name=r"Feagin's 10(8) pair"
        stages=20
    if ((file_name=='rk108curtis.txt')):
        name=r"Curtis' 10(8) pair"
        stages=21
    if ((file_name=='rk1210.txt')):
        name=r"Feagin's 12(10) pair"
        stages=25
    if ((file_name=='rk129hiroshi.txt')):
        name=r"Hiroshi's 12(9) pair"
        stages=29
    if ((file_name=='rk1412.txt')):
        name=r"Feagin's 14(12) pair"
        stages=35
    return name, stages

def load_rk_from_file(file_name='rk1210.txt', load_pair=True):
    name, stages = method_name_and_stages(file_name)
    f = open(os.path.join(path,file_name))
    b = np.zeros((stages))
    bhat = np.zeros((stages))
    A = np.zeros((stages,stages))
    c = np.zeros((stages))

    for line in f.readlines():
        words = line.split()
        if words == []:
            continue

        if words[-1]=='b[k]':
            current_array = 'b'
        elif words[-1]=='c[k]':
            current_array = 'c'
        elif words[-1]=='bhat[k]':
            current_array = 'bhat'
        elif words[-1]=='A[k,j]':
            current_array = 'A'

        try:
            k = int(words[0])
            if current_array == 'b':
                b[k] = float(words[1])
            elif current_array =='bhat':
                bhat[k] = float(words[1])
            elif current_array == 'A':
                j = int(words[1])
                A[k,j] = float(words[2])
        except ValueError:
            continue

    if load_pair:
        return rk.ExplicitRungeKuttaPair(A=A, b=b, bhat=bhat, name=name)
    else:
        return rk.ExplicitRungeKuttaMethod(A=A, b=b, name=name)
