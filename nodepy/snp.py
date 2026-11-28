"""
This module enables exact computation in NodePy by introducing functions
that understand numpy arrays of floats or symbolic objects.
"""
import sympy
import numpy as np

dtypes = {'exact': object, 'numeric': np.float64}

def normalize(*arrays):
    """
    For symbolic arrays, converts all non-symbolic entries to sympy types.
    """
    for array in arrays:
        if array is None: continue
        if array.dtype==object:
            onedarray=array.reshape(-1)
            for i,elem in enumerate(onedarray):
                if not isinstance(elem,sympy.Basic):
                    onedarray[i]=sympy.S(elem)
    if len(arrays)==1: return arrays[0]
    else: return arrays


class poly1d(np.poly1d):
    """
    A numpy.poly1d whose string form also works when the coefficients are
    SymPy objects.  (numpy.poly1d formats each coefficient with
    f'{c:.4g}', which SymPy numbers do not support.)

    Numerical coefficients are printed as floating-point numbers, like
    numpy.poly1d does; coefficients that contain symbols are printed by SymPy.
    """
    def __str__(self):
        if self.coeffs.dtype != object:
            return super(poly1d, self).__str__()
        try:
            coeffs = np.array([complex(c) for c in self.coeffs])
        except TypeError:   # Coefficients contain free symbols
            x = sympy.Symbol(self.variable)
            n = len(self.coeffs) - 1
            return str(sum(c*x**(n-k) for k, c in enumerate(self.coeffs)))
        if np.all(coeffs.imag == 0):
            coeffs = coeffs.real
        return str(np.poly1d(coeffs, variable=self.variable))


def is_zero(x):
    """
    True if x is zero.  Unlike x == 0, this is also True for a SymPy Float
    zero, which SymPy does not consider equal to the integer 0.
    """
    if isinstance(x, sympy.Basic):
        return x.is_zero is True
    return x == 0


def ones(n,mode='exact'):
    return normalize(np.ones(n,dtype=dtypes[mode]))

def eye(n,mode='exact'):
    return normalize(np.eye(n,dtype=dtypes[mode]))

def tri(n,mode='exact'):
    x = np.array(1*(np.tri(n)>0),dtype=dtypes[mode])
    x = normalize(x)
    return x

def solve(A,b):
    """Solve Ax = b.
        If A holds exact values, solve exactly using sympy.
        Otherwise, solve in floating-point using numpy.
    """
    if A.dtype==object:
        Asym=sympy.Matrix(A)
        bsym=sympy.Matrix(b)
        # Silly sympy makes row vectors when we want a column vector:
        if bsym.shape[0]==1:
            bsym = bsym.T
        if Asym.is_lower: # Take advantage of structure to solve quickly
            xsym=sympy.zeros(*b.shape)
            xsym = Asym.lower_triangular_solve(bsym)
        else: # This is slower:
            xsym = Asym.LUsolve(bsym)

        xsym = sympy.matrix2numpy(xsym)
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

def zeros(shape, mode='exact'):
    return normalize(np.zeros(np.array(shape).astype(int), dtype=dtypes[mode]))

def diag(v, k=0):
    return normalize(np.diag(v,k))

def poly(A,mode='exact'):
    if mode=='exact': return sympy.berkowitz

def norm(x):
    if x.dtype == object:
        from sympy import sqrt
        return sqrt(np.sum(x**2))
    else:
        return np.sqrt(float(np.sum(x**2)))

def array(x):
    return np.array(x,dtype=object)

def simplify(x):
    shape = x.shape
    x = list(map(sympy.simplify, x.reshape(-1)))
    return np.reshape(x,shape)

def printable(num,digits=3,return_zero=False,return_one=True):
    if num==0:
        if return_zero: return '0'
        else: return ''
    elif num==1 and return_one==False:
        return ''
    elif num==-1 and return_one==False:
        return '-'
    # Surprisingly, sympy does not handle these cases
    elif num == np.inf:
        return r'\infty'
    elif num == -np.inf:
        return r'-\infty'
    if isinstance(num,float):
        return '%.3f' % num
    else:
        from sympy.printing import latex
        return latex(num)
