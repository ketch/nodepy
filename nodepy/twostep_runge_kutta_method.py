r"""
Class for two-step Runge-Kutta methods, and various functions related to them.

This module is extremely experimental and some parts may be incompatible
with the rest of nodepy.

**Examples**::

    >>> from nodepy import twostep_runge_kutta_method as tsrk

* Load methods::

    >>> tsrk4 = tsrk.loadTSRK('order4')
    >>> tsrk5 = tsrk.loadTSRK('order5')

    >>> print(tsrk4)
    Two-step Runge-Kutta Method
    General
     -113/88      | 1435/352      -479/352      | 1
     -103/88      | 1917/352      -217/352      |               1
    ______________|_____________________________|_____________________________
    -4483/8011    | 180991/96132  -17777/32044  | -44709/32044  48803/96132
    >>> print(tsrk4.latex())
    \begin{align}
      \begin{array}{c|cc|cc}
      - \frac{113}{88} & \frac{1435}{352} & - \frac{479}{352} & 1 & 0\\
      - \frac{103}{88} & \frac{1917}{352} & - \frac{217}{352} & 0 & 1\\
      \hline
      - \frac{4483}{8011} & \frac{180991}{96132} & - \frac{17777}{32044} & - \frac{44709}{32044} & \frac{48803}{96132}
      \end{array}
    \end{align}

    >>> tsrk4.plot_stability_region()

* Check their order of accuracy::

    >>> tsrk4.order()
    4
    >>> tsrk5.order(tol=1.e-3)
    5

* Get the radius of absolute monotonicity::

    >>> tsrk4.absolute_monotonicity_radius()
    0
    >>> tsrk5.absolute_monotonicity_radius()
    0


**References**:
    [jackiewicz1995,butcher1997,hairer1997]

"""
from __future__ import print_function
from __future__ import division

from __future__ import absolute_import
import numpy as np

import nodepy.rooted_trees as rt
import nodepy.snp as snp
from nodepy.strmanip import *
from nodepy.general_linear_method import GeneralLinearMethod
from six.moves import range

#=====================================================
class TwoStepRungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    r""" General class for Two-step Runge-Kutta Methods
        The representation uses the form and partly the notation of :cite:`jackiewicz1995`,
        equation (1.3).

        `\begin{align*}
        y^n_j = & d_j u^{n-1} + (1-d_j)u^n + \Delta t \sum_{k=1}^{s}
        (\hat{a}_{jk} f(y_k^{n-1}) + a_{jk} f(y_k^n)) & (1\le j \le s) \\
        u^{n+1} = & \theta u^{n-1} + (1-\theta)u^n + \Delta t \sum_{j=1}^{s}(\hat{b}_j f(y_j^{n-1}) + b_j f(y_j^n))
        \end{align*}`
    """
    def __init__(self,d,theta,A,b,Ahat=None,bhat=None,type='General',name='Two-step Runge-Kutta Method'):
        r"""
            Initialize a 2-step Runge-Kutta method."""
        d,A,b,Ahat,bhat=snp.normalize(d,A,b,Ahat,bhat)
        self.s = max(np.shape(b))
        self.d,self.theta,self.A,self.b = d,theta,A,b
        self.Ahat,self.bhat=Ahat,bhat
        #if type=='General': self.Ahat,self.bhat=Ahat,bhat
        #elif type=='Type I':
        #    self.Ahat = np.zeros([self.s,self.s])
        #    self.bhat = np.zeros([self.s,1])
        #elif type=='Type II':
        #    self.Ahat = np.zeros([self.s,self.s])
        #    self.Ahat[:,0]=Ahat
        #    self.bhat = np.zeros([self.s,1])
        #    self.bhat[0]=bhat
        #else: raise Exception('Unrecognized type')
        self.name=name
        self.type=type

    def order(self,tol=1.e-13):
        r"""
            Return the order of a Two-step Runge-Kutta method.
            Computed by computing the elementary weights corresponding
            to the appropriate rooted trees.
        """
        p=0
        while True:
            z=self.order_conditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def __len__(self):
        """
            The length of the method is the number of stages.
        """
        return np.size(self.A,0)

    def order_conditions(self,p):
        r"""
            Evaluate the order conditions corresponding to rooted trees
            of order $p$.

            **Output**:
                - A vector $z$ of residuals (the amount by which each
                  order condition is violated)

            TODO: Implement simple order conditions (a la Albrecht)
                    for Type I & II TSRKs
        """
        from numpy import dot
        d,theta,Ahat,A,bhat,b=self.d,self.theta,self.Ahat,self.A,self.bhat,self.b
        e=np.ones(len(b))
        b=b.T; bhat=bhat.T
        c=dot(Ahat+A,e)-d
        code=TSRKOrderConditions(p)
        z=np.zeros(len(code))
        for i in range(len(code)):
            exec('yy='+code[i])
            exec('z[i]='+code[i])
        return z


    def stability_matrix(self,z):
        r"""
            Constructs the stability matrix of a two-step Runge-Kutta method.
            Right now just for a specific value of z.
            We ought to use Sage to do it symbolically.

            **Output**:
                M -- stability matrix evaluated at z

            WARNING: This only works for Type I & Type II methods
            right now!!!
        """
        s = self.Ahat.shape[1]
        if self.type == 'General':
            # J Y^n = K Y^{n-1}
            K1 = np.column_stack((z*self.Ahat,self.d,1-self.d))
            K2 = snp.zeros(s+2); K2[-1] = 1
            K3 = np.concatenate((z*self.bhat,np.array((self.theta,1-self.theta))))
            K = np.vstack((K1,K2,K3))

            J = snp.eye(s+2)
            J[:s,:s] = J[:s,:s] - z*self.A
            J[-1,:s] = z*self.b

            M = snp.solve(J.astype('complex64'),K.astype('complex64'))
            #M = snp.solve(J, K) # This version is slower

        else:
            D=np.hstack([1.-self.d,self.d])
            thet=np.hstack([1.-self.theta,self.theta])
            A,b=self.A,self.b
            if self.type=='Type II':
                ahat = np.zeros([self.s,1]); ahat[:,0] = self.Ahat[:,0]
                bh = np.zeros([1,1]); bh[0,0]=self.bhat[0]
                A = np.hstack([ahat,self.A])
                A = np.vstack([np.zeros([1,self.s+1]),A])
                b =  np.vstack([bh,self.b])

            M1=np.linalg.solve(np.eye(self.s)-z*self.A,D)
            L1=thet+z*np.dot(self.b.T,M1)
            M=np.vstack([L1,[1.,0.]])
        return M

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        import copy
        numself = copy.deepcopy(self)
        if self.A.dtype==object:
            for coeff_array in ['A','Ahat','b','bhat','theta','d']:
                setattr(numself,coeff_array,np.array(getattr(self,coeff_array),dtype=np.float64))
        return numself

    def plot_stability_region(self,N=50,bounds=None,
                    color='r',filled=True,scaled=False):
        r"""
            Plot the region of absolute stability
            of a Two-step Runge-Kutta method, i.e. the set

            `\{ z \in C : M(z) is power bounded \}`

            where $M(z)$ is the stability matrix of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        method = self.__num__() # Use floating-point coefficients for efficiency

        import matplotlib.pyplot as plt
        if bounds is None:
            from nodepy.utils import find_plot_bounds
            stable = lambda z : max(abs(np.linalg.eigvals(method.stability_matrix(z))))<=1.0
            bounds = find_plot_bounds(np.vectorize(stable),guess=(-10,1,-5,5))
            if np.min(np.abs(np.array(bounds)))<1.e-14:
                print('No stable region found; is this method zero-stable?')

        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j

        maxroot = lambda z : max(abs(np.linalg.eigvals(method.stability_matrix(z))))
        Mroot = np.vectorize(maxroot)
        R = Mroot(Z)

        if filled:
            plt.contourf(X,Y,R,[0,1],colors=color)
        else:
            plt.contour(X,Y,R,[0,1],colors=color)
        plt.title('Absolute Stability Region for '+self.name)
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.axis('Image')

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r"""
            Returns the radius of absolute monotonicity
            of a TSRK method.
        """
        from nodepy.utils import bisect
        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        return r

    def latex(self):
        """A laTeX representation of the compact form."""
        from sympy.printing import latex

        d       = self.d
        A       = self.A
        Ahat    = self.Ahat
        b       = self.b
        bhat    = self.bhat
        theta   = self.theta

        s= r'\begin{align}'
        s+='\n'
        s+=r'  \begin{array}{c|'
        s+='c'*(len(self)) +'|'
        s+='c'*(len(self))
        s+='}\n'
        for i in range(len(self)):
            s+='  '+latex(d[i])

            for j in range(len(self)):
                s+=' & '+latex(Ahat[i,j])

            for j in range(len(self)):
                s+=' & '+latex(A[i,j])

            s+=r'\\'
            s+='\n'
        s+=r'  \hline'
        s+='\n'
        s+= '  '+latex(theta)
        for j in range(len(self)):
            s+=' & '+latex(bhat[j])
        for j in range(len(self)):
            s+=' & '+latex(b[j])
        s+='\n'
        s+=r'  \end{array}'
        s+='\n'
        s+=r'\end{align}'
        s=s.replace('- -','')
        return s


    def __str__(self):
        from nodepy.utils import array2strings
        from nodepy.runge_kutta_method import _get_column_widths

        d       = array2strings(self.d)
        A       = array2strings(self.A)
        Ahat    = array2strings(self.Ahat)
        b       = array2strings(self.b)
        bhat    = array2strings(self.bhat)

        theta = str(self.theta)
        lenmax, colmax = _get_column_widths([d, Ahat, A, bhat, b])

        s   = self.name+'\n'+self.type+'\n'
        for i in range(len(self)):
                s+=d[i].ljust(colmax+1)+'|'
                for j in range(len(self)):
                        s+=Ahat[i,j].ljust(colmax+1)
                s+=' |'
                for j in range(len(self)):
                        s+=A[i,j].ljust(colmax+1)
                s=s.rstrip()+'\n'
        s+='_'*(colmax+1)+('|_'+'_'*(colmax+1)*np.size(A,0))*2+'\n'

        s+= theta.ljust(colmax)
        s+=' |'
        for j in range(len(self)):
                s+=bhat[j].ljust(colmax+1)
        s+=' |'
        for j in range(len(self)):
                s+=b[j].ljust(colmax+1)
        return s.rstrip()

    def spijker_form(self):
        r""" Returns arrays $S,T$ such that the TSRK can be written
            $$ w = S x + T f(w),$$
            and such that $\[S \ \ T\]$ has no two rows equal.
            See the TSRK paper by Ketcheson, Gottlieb, and Macdonald.
            Equation (2.5) therein gives the Spijker form for general TSRKs,
            while the last (unnumbered) equation of Section 4.2 gives the
            form for TSRKs of Type II (which are the type described by (4.2)).
        """
        s=self.s
        if self.type=='General':
            zero_column = np.zeros( (s,1) )
            T3 = np.hstack( (self.Ahat,zero_column,self.A,zero_column) )
            T4 = np.hstack( (self.bhat, 0, self.b, 0) )
            T = np.vstack( (np.zeros([s+1,2*s+2]),T3,T4) )

            S1 = np.hstack( (np.zeros([s+1,1]),np.eye(s+1)) )
            S2 = np.column_stack( (self.d,np.zeros([s,s]),1-self.d) )
            S3 = np.hstack( ([[self.theta]],np.zeros([1,s]),[[1-self.theta]]) )
            S = np.vstack( (S1,S2,S3) )

        elif self.type=='Type I':
            K = np.vstack([self.A,self.b.T])
            T2 = np.hstack([np.zeros([s+1,1]),K,np.zeros([s+1,1])])
            T =  np.vstack([np.zeros([1,s+1]),T2])
            S1 = np.vstack([np.zeros([1,1]),self.d,np.array([[self.theta]])])
            S2 = np.vstack([np.zeros([1,1]),1-self.d,np.array([[1-self.theta]])])
            S = np.hstack([S1,S2])

        elif self.type=='Type II':
            S0 = np.array([1,0])
            S1 = np.hstack([self.d,1-self.d])
            S2 = np.array([self.theta,1-self.theta])
            S =  np.vstack([S0,S1,S2])
            ahat = np.zeros([s,1])
            ahat[:,0] = self.Ahat[:,0]
            bh = np.zeros([1,1])
            bh[0,0]=self.bhat[0]
            T0 = np.zeros([1,s+2])
            T1 =  np.hstack([ahat,self.A,np.zeros([s,1])])
            T2 =  np.hstack([bh,self.b.T,np.zeros([1,1])])
            T = np.vstack([T0,T1,T2])

        return S.astype(np.float64), T.astype(np.float64)


    def is_absolutely_monotonic(self,r,tol):
        r""" Returns 1 if the TSRK method is absolutely monotonic
            at $z=-r$.

            The method is absolutely monotonic if $(I+rT)^{-1}$ exists
            and
            $$(I+rT)^{-1}T \\ge 0$$
            $$(I+rT)^{-1}S \\ge 0$$

            The inequalities are interpreted componentwise.

            **References**:
                #. [spijker2007]
        """
        S, T = self.spijker_form()
        m = np.shape(T)[0]
        X = np.eye(m) + r * T
        if abs(np.linalg.det(X)) < tol:
            return 0
        P = np.linalg.solve(X, T)
        R = np.linalg.solve(X, S)
        if P.min()<-tol or R.min()<-tol:
            return 0
        else:
            return 1
        # Need an exception here if rhi==rmax


#================================================================
# Functions for analyzing Two-step Runge-Kutta order conditions
#================================================================

def TSRKOrderConditions(p,ind='all'):
    from nodepy.rooted_trees import Emap_str
    forest=rt.list_trees(p)
    code=[]
    for tree in forest:
        code.append(tsrk_elementary_weight_str(tree)+'-'+Emap_str(tree))
        code[-1]=code[-1].replace('--','')
        code[-1]=code[-1].replace('1 ','e ')
        code[-1]=code[-1].replace('1)','e)')
    return code

def tsrk_elementary_weight(tree):
    """
        Constructs Butcher's elementary weights
        for Two-step Runge-Kutta methods
    """
    from sympy import Symbol
    bhat,b,theta=Symbol('bhat',False),Symbol('b',False),Symbol('theta',False)
    ew=bhat*tree.Gprod(rt.Emap,rt.Gprod,betaargs=[TSRKeta,Dmap],alphaargs=[-1])+b*tree.Gprod(TSRKeta,rt.Dmap)+theta*tree.Emap(-1)
    return ew

def tsrk_elementary_weight_str(tree):
    """
        Constructs Butcher's elementary weights
        for Two-step Runge-Kutta methods
        as numpy-executable strings
    """
    from nodepy.rooted_trees import Dmap_str, Emap_str
    ewstr='dot(bhat,'+tree.Gprod_str(rt.Emap_str,rt.Gprod_str,betaargs=[TSRKeta_str,Dmap_str],alphaargs=[-1])+')+dot(b,'+tree.Gprod_str(TSRKeta_str,rt.Dmap_str)+')+theta*'+Emap_str(tree,-1)
    ewstr=mysimp(ewstr)
    return ewstr

def TSRKeta(tree):
    from nodepy.rooted_trees import Dprod
    from sympy import symbols
    raise Exception('This function does not work correctly; use the _str version')
    if tree=='':  return 1
    if tree=='T': return symbols('c',commutative=False)
    return symbols('d',commutative=False)*tree.Emap(-1)+symbols('Ahat',commutative=False)*tree.Gprod(Emap,Dprod,betaargs=[TSRKeta],alphaargs=[-1])+symbols('A',commutative=False)*Dprod(tree,TSRKeta)

def TSRKeta_str(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods
    """
    from nodepy.rooted_trees import Dprod_str, Emap_str
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return '(d*'+Emap_str(tree,-1)+'+dot(Ahat,'+tree.Gprod_str(Emap_str,Dprod_str,betaargs=[TSRKeta_str],alphaargs=[-1])+')'+'+dot(A,'+Dprod_str(tree,TSRKeta_str)+'))'


#================================================================

def loadTSRK(which='All'):
    r"""
        Load two particular TSRK methods (From :cite:`jackiewicz1995`).

        The method of order five satisfies the order conditions only
        to four or five digits of accuracy.
    """
    from sympy import Rational
    one  = Rational(1,1)

    TSRK={}
    #================================================
    c = one
    lamda = 0
    theta = (6*c**2 - 12*c + 5)/(1 - 6*c**2)
    d    = np.array( [ (2*c - c**2 -2*lamda)/(2*c - 1) ] )
    Ahat = np.array( [[(c+c**2-lamda-2*c*lamda)/(2*c - 1)]] )
    A    = np.array( [[lamda]] )
    bhat = np.array( [2*(1-3*c**2)/(1-6*c**2)] )
    b    = np.array( [2*(2-6*c+3*c**2)/(1-6*c**2)] )
    TSRK['order3']=TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat,type='General')
    #================================================
    d=np.array([-113*one/88,-103*one/88])
    theta=-4483*one/8011
    Ahat=np.array([[1435*one/352,-479*one/352],[1917*one/352,-217*one/352]])
    A=np.eye(2,dtype=object)
    bhat=np.array([180991*one/96132,-17777*one/32044])
    b=np.array([-44709*one/32044,48803*one/96132])
    TSRK['order4']=TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat,type='General')
    #================================================
    d=np.array([-0.210299,-0.0995138])
    theta=-0.186912
    Ahat=np.array([[1.97944,0.0387917],[2.5617,2.3738]])
    A=np.zeros([2,2])
    bhat=np.array([1.45338,0.248242])
    b=np.array([-0.812426,-0.0761097])
    TSRK['order5']=TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat,type='General')
    if which=='All': return TSRK
    else: return TSRK[which]


def load_type2_TSRK(s,p,type='Type II'):
    r"""
    Load a TSRK method from its coefficients in an ASCII file
    (usually from MATLAB).  The coefficients are stored in the
    following order: [s d theta A b Ahat bhat].
    """
    path='/Users/ketch/Research/Projects/MRK/methods/TSRK/'
    file=path+'explicitzr'+str(s)+str(p)+'.tsrk'
    f=open(file,'r')
    coeff=[]
    for line in f:
        for word in line.split():
            coeff.append(float(word))

    s=int(coeff[0])
    A=np.zeros([s,s])
    Ahat=np.zeros([s,s])
    b=np.zeros([s,1])
    bhat=np.zeros([s,1])

    d=np.array(coeff[1:s+1],ndmin=2).T
    theta = coeff[s+1]
    for row in range(s):
        A[row,:] = coeff[s+2+s*row:s+2+s*(row+1)]
    b = np.array(coeff[s**2+s+2:s**2+2*s+2],ndmin=2).T
    for row in range(s):
        Ahat[row,:] = coeff[s**2+2*s+2+s*row:s**2+2*s+2+s*(row+1)]
    bhat = np.array(coeff[2*s**2+2*s+2:2*s**2+3*s+2],ndmin=2).T
    return TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat,type=type)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
