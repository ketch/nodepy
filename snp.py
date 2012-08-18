"""
This module enables exact computation in NodePy by introducing functions
that understand numpy arrays of floats or symbolic objects.
"""
import sympy
import numpy as np

dtypes={'exact':object,'numeric':np.float64}

def normalize(*arrays):
    """
    For symbolic arrays, converts all non-symbolic entries to sympy Rationals.
    """
    for array in arrays:
        if array==None: continue
        if array.dtype==object:
            onedarray=array.reshape(-1)
            for i,elem in enumerate(onedarray):
                if not isinstance(elem,sympy.basic.Basic):
                    onedarray[i]=sympy.Rational(elem)
    if len(arrays)==1: return arrays[0]
    else: return arrays


def eye(n,mode='exact'):
    return normalize(np.eye(n,dtype=dtypes[mode]))

def solve(A,b):
    if A.dtype==object:
        Asym=sympy.matrices.Matrix(A)
        bsym=sympy.matrices.Matrix(b)
        xsym = Asym.LUsolve(bsym)
        xsym = np.array(list(xsym),dtype=object)
        if len(b.shape)>1:
            shape = [A.shape[1],b.shape[0]]
        else:
            shape = [A.shape[1]]
        return np.reshape(xsym,shape)
    else:
        return np.linalg.solve(A,b)

def linspace(start,stop,num=50,endpoint=True,retstep=False):
    "This doesn't generally work as desired."
    return normalize(np.linspace(start,stop,num,endpoint,retstep))

def arange(start,stop=None,step=None,mode='exact'):
    return normalize(np.arange(start,stop,step,dtype=dtypes[mode]))

def zeros(shape,mode='exact'):
    return normalize(np.zeros(shape,dtype=dtypes[mode]))

def diag(x):
    return normalize(np.diag(x))

def poly(A,mode='exact'):
    if mode=='exact': return sympy.berkowitz

def array(x):
    return np.array(x,dtype=object)
