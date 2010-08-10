"""
Class for rooted trees, and various functions on rooted trees.

These are useful for analyzing the order conditions of multistage
numerical ODE solvers, such as Runge-Kutta methods and general
linear methods.

AUTHOR: David Ketcheson (08-29-2008)

**Examples**::

    >>> from rooted_trees import *
    >>> tree=RootedTree('{T^2{T{T}}{T}}')
    >>> tree.plot()
    >>> tree.order()
    9
    >>> tree.density()
    144
    >>> tree.symmetry()
    2

Topologically equivalent trees are considered equal::

    >>> tree2=RootedTree('{T^2{T}{T{T}}}')
    >>> tree2==tree
    True

"""
import numpy as np
import pylab as pl
from sympy import Symbol, factorial, sympify, Rational
#from sage.combinat.combinat import permutations
from nodepy_utils import permutations
from strmanip import *

#=====================================================
class RootedTree(str):
#=====================================================
    """
        The trees are represented as strings, using one of the notations
        introduced by Butcher (the third column of Table 300(I) of
        Butcher's text).  We use 'T' for tau (to represent a vertex)
        and '{}' for '[]' (to indicate that everything inside the braces
        is joined to a single parent node).  Thus the first four trees are:

        'T', '{T}', '{T^2}', {{T}}'
        
        A vertex with no children is referred to as a leaf.  
        The only convention assumed in the code is that at each level, 
        leaves are listed first (before more complex subtrees), 
        and if there are n leaves, we write 'T^n'.
        Currently, powers cannot be used for subtrees; thus

        '{{T}{T}}'

        is valid, while

        '{{T}^2}' 

        is not.  This may change in the future.

        **References**:  
            #. [butcher2003]_
            #. [hairer1993]_
    """
    def __init__(self,strg):
        """
            TODO:   - Check validity of strg more extensively
                    - Accept any leaf ordering, but convert it to our convention
                    - convention for ordering of subtrees?
        """
        if any([strg[i] not in '{}T^1234567890' for i in range(len(strg))]):
            raise TreeError(strg,
                'Not a valid rooted tree string (illegal character)')
        op,cl=strg.count('{'),strg.count('}')
        if op!=cl or (op+cl>0 and (strg[0]!='{' or strg[-1]!='}')):
            raise TreeError(strg,'Not a valid rooted tree string')
        self=strg

    def order(self):
        """
        Returns the order r(t) of the rooted tree.

        The order is the number of vertices in the tree.
        """
        if self=='T': return 1
        if self=='':  return 0
        r=self.count('{')
        pos=0
        while pos!=-1:
             pos=self.find('T',pos+1)
             if pos!=-1:
                  try: r+=int(self[pos+2])
                  except: r+=1
        return r

    def density(self):
        """
        Returns the density gamma(t) of the rooted tree.
        Density is the product of the orders of the subtrees.
        **Reference**: 
            - [butcher2003]_ p. 127, eq. 301(c)
        """
        gamma=self.order()
        nleaves,subtrees=self.parse_subtrees()
        for tree in subtrees:
             gamma*=tree.density()
        return gamma

    def symmetry(self):
        """ 
        Returns the symmetry sigma(t) of a rooted tree 
        **Reference**: 
            - [butcher2003]_ p. 127, eq. 301(b)
        """
        if self=='T': return 1
        sigma=1
        if self[1]=='T':
             try: sigma=factorial(int(self[3]))
             except: pass
        nleaves,subtrees=self.parse_subtrees()
        while len(subtrees)>0:
             st=subtrees[0]
             nst=subtrees.count(st)
             sigma*=factorial(nst)*st.symmetry()**nst
             while st in subtrees: subtrees.remove(st)
        return sigma

    def Emap(self,a=1):
        """ 
        Butcher's function E^a(t).
        Gives the B-series for the exact solution advanced 'a' steps
        in time.
        **Reference**: 
            [butcher1997]_
        """
        return Rational(a**self.order(),(self.density()))

    def Dmap(self):
        """ 
        Butcher's function D(t).  Represents differentiation.
        Defined by D(t)=0 except for D('T')=1.

        **Reference**: 
            #. [butcher1997]_
        """
        return self=='T'

    def lamda(self,alpha,extraargs=None):
        r""" 
            Computes Butcher's functional lambda on a given tree
            for the function alpha.  This is used to compute the
            product of two functions on trees.

            INPUT:
                alpha -- a function on rooted trees
                extraargs -- a string containing any additional arguments 
                                that must be passed to alpha

            OUTPUT:
                tprod -- a list of trees [t1, t2, ...]
                fprod -- a list of numbers [a1, a2, ...]

            The meaning of the output is that 
            \lambda(\alpha,t)(\beta)=a1*\beta(t1)+a2*\beta(t2)+... 

            **Reference**: 
                [butcher2003]_ pp. 275-276
        """
        if self=='': return [RootedTree('')],[0]
        if self=='T': return [RootedTree('T')],[1]
        t,u=self.factor()
        if extraargs:
            exec('l1,f1=t.lamda(alpha,'+str(extraargs)+')')
            exec('l2,f2=u.lamda(alpha,'+str(extraargs)+')')
            exec('alphau=alpha(u,'+str(extraargs)+')')
        else:
            l1,f1=t.lamda(alpha)
            l2,f2=u.lamda(alpha)
            alphau=alpha(u)
        tprod=l1
        fprod=[alphau*f1i for f1i in f1 if f1!=0]
        #FOIL:
        for i in range(len(l1)):
            if f1!=0:
                for j in range(len(l2)):
                    if f2!=0:
                        tprod.append(l1[i]*l2[j])
                        fprod.append(f1[i]*f2[j])
        return tprod,fprod

    def lamda_str(self,alpha,extraargs=None):
        """
            Alternate version of lamda above, but returns a string.
            Hopefully we can get rid of this (and the other str functions)
            when SAGE can handle noncommutative symbolic algebra.
        """
        if self=='': return [RootedTree('')],[0]
        if self=='T': return [RootedTree('T')],[1]
        t,u=self.factor()
        if extraargs:
            exec('l1,f1=t.lamda_str(alpha,'+str(extraargs)+')')
            exec('l2,f2=u.lamda_str(alpha,'+str(extraargs)+')')
            exec('alphau=alpha(u,'+str(extraargs)+')')
        else:
            l1,f1=t.lamda_str(alpha)
            l2,f2=u.lamda_str(alpha)
            alphau=alpha(u)
        tprod=l1
        fprod=[str(f1i)+'*'+alphau for f1i in f1 if f1!=0]
        #FOIL:
        for i in range(len(l1)):
            if f1!=0:
                for j in range(len(l2)):
                    if f2!=0:
                        tprod.append(l1[i]*l2[j])
                        fprod.append(str(f1[i])+'*'+str(f2[j]))
        return tprod,fprod

    def factor(self):
        """ 
            Returns two rooted trees, t and u, such that self=t*u.

            **Input**: 
                - self -- any rooted tree
            **Output**: 
                - t,u -- a pair of rooted trees whose product t*u is equal to self.

            **Examples**::

                >>> tree=RootedTree('{T^2{T}}')
                >>> t,u=tree.factor()
                >>> t
                '{T{T}}'
                >>> u
                'T'
                >>> t*u==tree
                True

            .. note:: This function is typically only called by lamda().
        """
        nleaves,subtrees=self.parse_subtrees()
        if nleaves==0: # Root has no leaves
            t=RootedTree('{'+''.join(subtrees[1:])+'}')
            u=RootedTree(subtrees[0])
        if nleaves==1:
             t=RootedTree(self[0]+self[2:])
             u=RootedTree('T')
        if nleaves==2:
             t=RootedTree(self[0:2]+self[4:])
             u=RootedTree('T')
        if nleaves>2:
             t=RootedTree(self[0:3]+str(int(self[3])-1)+self[4:])
             u=RootedTree('T')
        if t=='{}': t=RootedTree('T')
        return t,u

    def Gprod(self,alpha,beta,alphaargs='',betaargs=''):
        r""" 
            Returns the product of two functions on a given tree.

            INPUT:
                alpha, beta -- two functions on rooted trees
                               that return symbolic or numeric values
                alphaargs   -- a string containing any additional arguments 
                                that must be passed to function alpha
                betaargs    -- a string containing any additional arguments 
                                that must be passed to function beta

            OUTPUT:
                (alpha*beta)(self)
                i.e., the function that is the product (in G) of the
                functions alpha and beta.  Note that this product is
                not commutative.

            The product is given by

            (\alpha*\beta)('')=\beta('')
            (\alpha*\beta)(t) = \lambda(alpha,t)(\beta) + \alpha(t)\beta('')

            .. note::
                Gprod can be used to compute products of more than two
                functions by passing Gprod itself in as beta, and providing
                the remaining functions to be multiplied as betaargs.

            **Reference**: [butcher2003]_ p. 276, Thm. 386A 
        """
        exec('trees,facs=self.lamda(alpha,'+alphaargs+')')
        s=0
        for i in range(len(trees)):
            exec('s+=facs[i]*beta(trees[i],'+betaargs+')')
        exec('s+=alpha(self,'+alphaargs+')*beta(RootedTree(""),'+betaargs+')')
        return s

    def Gprod_str(self,alpha,beta,alphaargs='',betaargs=''):
        """ 
            Alternate version of Gprod, but operates on strings.
            Hopefully can be eliminated later in favor of symbolic
            manipulation.
        """
        exec('trees,facs=self.lamda_str(alpha,'+alphaargs+')')
        s=""
        for i in range(len(trees)):
            if facs[i]!=0:
                exec('bet=beta(trees[i],'+betaargs+')')
                if bet not in ['0','']:
                    if i>0: s+='+'
                    s+=str(facs[i])+"*"+bet
        exec('bet=beta(RootedTree(""),'+betaargs+')')
        if bet not in ['0','']:
            exec('alph=alpha(self,'+alphaargs+')')
            s+="+"+str(sympify(alph+'*'+bet))
        return s

    def plot(self,nrows=1,ncols=1,iplot=1,ttitle=''):
        """
            Plots the rooted tree.

            INPUT: (optional)
                nrows, ncols -- number of rows and columns of subplots
                                in the figure
                iplot        -- index of the subplot in which to plot
                                this tree

                These are only necessary if plotting more than one tree
                in a single figure using subplot.

            OUTPUT: returns nothing, but plots the tree.

            The plot is created recursively by
            plotting the root, parsing the subtrees, plotting the 
            subtrees' roots, and calling plot_subtree on each child
        """
        if iplot==1: pl.clf()
        pl.subplot(nrows,ncols,iplot)
        pl.hold(True)
        pl.scatter([0],[0])
        if self!='T': self.plot_subtree(0,0,1.)

        fs=int(pl.ceil(30./nrows))
        pl.title(ttitle,{'fontsize': fs})
        pl.xticks([])
        pl.yticks([])
        pl.hold(False)
        pl.axis('off')
        #pl.show()
        #pl.ioff()


    def plot_subtree(self,xroot,yroot,xwidth):
        """
            Recursively plots subtrees.  Should only be called from plot().

            INPUT:
                xroot, yroot -- coordinates at which root of this subtree 
                                is plotted
                xwidth -- width in which this subtree must fit, in order
                            to avoid possibly overlapping with others
        """
        ychild=yroot+1
        nleaves,subtrees=self.parse_subtrees()
        nchildren=nleaves+len(subtrees)

        dist=xwidth*(nchildren-1)/2.
        xchild=np.linspace(xroot-dist,xroot+dist,nchildren)
        pl.scatter(xchild,ychild*np.ones(nchildren))
        for i in range(nchildren):
            pl.plot([xroot,xchild[i]],[yroot,ychild],'-k')
            if i>nleaves-1:
                    subtrees[i-nleaves].plot_subtree(xchild[i],ychild,xwidth/3.)


    def parse_subtrees(self):
        """ 
            Returns the number of leaves and a list of the subtrees,
            for a given rooted tree.

            OUTPUT:
                nleaves  -- number of leaves attached directly to the root
                subtrees -- list of non-leaf subtrees attached to the root

            The method can be thought of return what remains if the
            root of the tree is removed.  For efficiency, instead of
            returning possibly many copies of 'T', the leaves are just
            returned as a number.
        """
        if str(self)=='T' or str(self)=='': return 0,[]
        pos=0
        #Count leaves at current level
        if self[1]=='T':    
            if self[2]=='^': nleaves=int(self[3])
            else: nleaves=1
        else: nleaves=0

        subtrees=[]
        while pos!=-1:
            pos=self.find('{',pos+1)
            if pos!=-1:
                subtrees.append(RootedTree(get_substring(self,pos)))
                pos=open_to_close(self,pos)

        return nleaves,subtrees

    def all_equivalent_trees(self):
        """ 
            Returns a list of all strings (subject to our assumptions)
            equivalent to a given tree 

            INPUT: 
                self     -- any rooted tree
            OUTPUT:  
                treelist -- a list of all the 'legal' tree strings
                            that produce the same tree.

            The list of equivalent trees is obtained by taking all
            permutations of the (non-leaf) subtrees.
            This routine is used to test equality of trees.
        """
        nleaves,subtrees=self.parse_subtrees()
        if len(subtrees)==0: return [self]
        for i in range(len(subtrees)): subtrees[i]=str(subtrees[i])
        treelist = [RootedTree('{'+powerString('T',nleaves,powchar='^')+
                    ''.join(sts)+'}') for sts in permutations(subtrees)]
        return treelist

    def __eq__(self,tree2):
        """
            Test equivalence of two rooted trees.
            Generates all 'legal' strings equivalent to the first
            tree, and checks whether the second is in that list.
        """
        ts=[str(t) for t in self.all_equivalent_trees()]
        if str(tree2) in ts: return True
        else: return False

    def __mul__(self,tree2):
        """ 
            Returns Butcher's product: t*u is the tree obtained by
            attaching the root of u as a child to the root of t. 
        """
        if self=='T': return RootedTree('{'+tree2+'}')
        if tree2=='T':  # We're just adding a leaf to self
            nleaves,subtrees=self.parse_subtrees()
            if nleaves==0: return RootedTree(self[0]+'T'+self[1:])
            if nleaves==1: return RootedTree(self[0]+'T^2'+self[2:])
            if nleaves>1:
                return RootedTree(self[0:3]+str(int(self[3])+1)+self[4:])
        else: return RootedTree(self[:-1]+tree2+'}') # tree2 wasn't just 'T'

    def elementary_weight(self,level=0):
        """ Compute Butcher's elementary weight for a given 
             rooted tree 
             
             This function is DEPRECATED.
             Use RK_elementary_weight() instead.
        """

        if level==0: 
            elwt='np.dot(b'
            csym='c'
        else: 
            elwt='np.dot(A'
            csym='D'
        nleaves,subtrees=self.parse_subtrees()
        if len(subtrees)==0: 
            elwt+=',c'+('*c'*(nleaves-1))+')'
            return elwt
        elwt+=('*'+csym)*nleaves+','
        for isubtree in range(len(subtrees)):
            if isubtree>0: elwt+='*'
            elwt+=subtrees[isubtree].elementary_weight(level+1)
        return elwt+')'
#=====================================================
#End of RootedTree class
#=====================================================

#=====================================================
class TreeError(Exception):
#=====================================================
    """
        Exception class for rooted tree exceptions.
    """
    def __init__(self,tree,msg=''):
        self.tree=tree
        self.msg=msg

    def __str__(self):
        return self.msg+': '+self.tree
#=====================================================
#End of TreeException class
#=====================================================

#=====================================================
def plot_all_trees(p):
#=====================================================
    """ Plots all rooted trees of order p """
    from twostep_runge_kutta_method import tsrk_elementary_weight_str

    forest=recursive_trees(p)
    nplots=len(forest)
    nrows=int(np.ceil(np.sqrt(float(nplots))))
    ncols=int(np.floor(np.sqrt(float(nplots))))
    if nrows*ncols<nplots: ncols=ncols+1
    for tree in forest:
        #ttitle=tsrk_elementary_weight_str(tree)+'-1/'+str(tree.density())
        ttitle=tree
        tree.plot(nrows,ncols,forest.index(tree)+1,ttitle=ttitle)
    fig=pl.figure(1)
    pl.setp(fig,facecolor='white')

#=====================================================
def recursive_trees(p,ind='all'):
#=====================================================
  """ 
    Returns rooted trees of order p.

    INPUT: 
        p   -- order of trees desired
        ind -- if given, returns a single tree corresponding to this index.
                Not very useful since the ordering isn't obvious.
                
    OUTPUT: list of all trees of order p (or just one, if ind is provided).
    
    Generates the rooted trees using Albrecht's 'Recursion 3'.

    **Examples**:

    Produce column of Butcher's Table 302(I)::

        >>> for i in range(1,11):
        >>>     forest=recursive_trees(i)
        >>>     print len(forest)
        1
        1
        2
        4
        9
        20
        48
        115
        286
        719

    .. note:: Generates all trees up to order at least 10.  Above some order
            it will not get them all (need to code more subloops).

    TODO: Implement Butcher's formula (Theorem 302B) for the number
            of trees and determine to what order this is valid.

    **Reference**: [albrecht1996]_
  """

  if p==0: return [RootedTree('')]
  W=[[],[]] #This way indices agree with Albrecht
  R=[[],[]]
  R.append([RootedTree("{T}")])
  W.append([RootedTree("{{T}}")])
  for i in range(3,p):
     #Construct R[i]
     ps=powerString("T",i-1,powchar="^")
     R.append([RootedTree("{"+ps+"}")])
     for w in W[i-1]:
        R[i].append(w)
     #Construct W[i]
     #l=0:
     W.append([RootedTree("{"+R[i][0]+"}")])
     for r in R[i][1:]:
        W[i].append(RootedTree("{"+r+"}"))
     for l in range(1,i-1): #level 1
        for r in R[i-l]:
          ps=powerString("T",l,powchar="^")
          W[i].append(RootedTree("{"+ps+r+"}"))
     for l in range(0,i-3): #level 2
        for n in range(2,i-l-1):
          m=i-n-l
          if m<=n: #Avoid duplicate conditions
             for Rm in R[m]:
                lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                for Rn in R[n][lowlim:]:
                  ps=powerString("T",l,powchar="^")
                  W[i].append(RootedTree("{"+ps+Rm+Rn+"}"))
     for l in range(0,i-5): #level 3
        for n in range(2,i-l-3):
          for m in range(2,i-l-n-1):
             s=i-m-n-l
             if m<=n and n<=s: #Avoid duplicate conditions
                for Rm in R[m]:
                  lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                  for Rn in R[n][lowlim:]:
                     lowlim2=(n<s and [0] or [R[n].index(Rn)])[0]
                     for Rs in R[s][lowlim2:]:
                        ps=powerString("T",l,powchar="^")
                        W[i].append(RootedTree("{"+ps+Rm+Rn+Rs+"}"))
     for l in range(0,i-7): #level 4
        for n in range(2,i-l-5):
          for m in range(2,i-l-n-3):
             for s in range(2,i-l-n-m-1):
                t=i-s-m-n-l
                if s<=t and n<=s and m<=n: #Avoid duplicate conditions
                  for Rm in R[m]:
                     lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                     for Rn in R[n][lowlim:]:
                        lowlim2=(n<s and [0] or [R[n].index(Rn)])[0]
                        for Rs in R[s][lowlim2:]:
                          lowlim3=(s<t and [0] or [R[s].index(Rs)])[0]
                          for Rt in R[t]:
                             ps=powerString("T",l,powchar="^")
                             W[i].append(RootedTree("{"+ps+Rm+Rn+Rs+Rt+"}"))
  # The recursion above generates all trees except the 'blooms'
  # Now add the blooms:
  W[0].append(RootedTree("T"))
  for i in range(1,p):
     ps=powerString("T",i,powchar="^")
     W[i].append(RootedTree("{"+ps+"}"))

  if ind=='all': return W[p-1]
  else: return W[p-1][ind]


def powerString(s,pow,powchar="**",trailchar=''):
  if pow==0:
     return ""
  else:
     if pow==1:
        return s+trailchar
     else:
        return s+powchar+str(pow)+trailchar


#=====================================================
# Functions on trees
#=====================================================

def Dprod(tree,alpha):
    """
    Evaluate (alpha*D)(t).  Note that this is not equal to (D*alpha)(t).
    This function is necessary (rather than just using Gprod)
    in order to avoid infinite recursions.
    """
    if tree=='': return 0
    if tree=='T': return alpha(RootedTree(''))
    nleaves,subtrees=tree.parse_subtrees()
    result=alpha(RootedTree('T'))**nleaves
    for subtree in subtrees:
        result*=alpha(subtree)
    return result

def Dprod_str(tree,alpha):
    if tree=='': return '0'
    if tree=='T': return alpha(RootedTree(''))
    nleaves,subtrees=tree.parse_subtrees()
    result=powerString(alpha(RootedTree('T')),nleaves)
    for subtree in subtrees:
        if result!='': result+='*'
        result+=alpha(subtree)
    return result

def Dmap(tree):
    """ 
    Butcher's function D(t).  Represents differentiation.
    Defined by D(t)=0 except for D('T')=1.
    """
    return tree=='T'

def Dmap_str(tree):
    return str(int(tree=='T'))

def Gprod(tree,alpha,beta,alphaargs='',betaargs=''):
    """ Returns the product of two functions on a given tree.
         See Butcher p. 276, Thm. 386A """
    return tree.Gprod(alpha,beta,alphaargs,betaargs)

def Gprod_str(tree,alpha,beta,alphaargs='',betaargs=''):
    return tree.Gprod_str(alpha,beta,alphaargs,betaargs)

def Emap(tree,a=1):
    """ 
    Butcher's function E^a(t).
    Gives the B-series for the exact solution advanced 'a' steps
    in time.
    """
    return Rational(a**tree.order(),(tree.density()))

def Emap_str(tree,a=1):
    return str(Rational(a**tree.order(),(tree.density())))

#=====================================================
#Runge-Kutta functions
#=====================================================
def RKeta(tree):
    if tree=='':  return Symbol('e',False)
    if tree=='T': return Symbol('c',False)
    return Symbol('A',False)*Dprod(tree,RKeta)

def RKeta_str(tree):
    """
    Computes eta(t) for Runge-Kutta methods
    """
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return 'dot(A,'+Dprod_str(tree,RKeta_str)+')'

#=====================================================
#Two-step Runge-Kutta functions
#=====================================================
def TSRKeta(tree):
    if tree=='':  return 1
    if tree=='T': return Symbol('c',False)
    return Symbol('d',False)*tree.Emap(-1)+Symbol('Ahat',False)*tree.Gprod(Emap,Dprod,betaargs='TSRKeta',alphaargs='-1')+Symbol('A',False)*Dprod(tree,TSRKeta)

def TSRKeta_str(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods
    """
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return '(d*'+str(tree.Emap(-1))+'+dot(Ahat,'+tree.Gprod_str(Emap_str,Dprod_str,betaargs='TSRKeta_str',alphaargs='-1')+')'+'+dot(A,'+Dprod_str(tree,TSRKeta_str)+'))'

#=====================================================
#Three-step Runge-Kutta functions ???
#=====================================================
def ThSRKeta(tree):
    if tree=='':  return 1
    if tree=='T': return Symbol('c',False)
    return Symbol('d2',False)*tree.Emap(-1)+Symbol('d3',False)*tree.Emap(-2)+Symbol('A',False)*Dprod(tree,ThSRKeta)

def ThSRKeta_str(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods -- Python string
    """
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return '(d2*'+str(tree.Emap(-1))+'+(d3*'+str(tree.Emap(-2))+'+dot(A,'+Dprod_str(tree,ThSRKeta_str)+'))'

def ThSRKeta_str_matlab(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods -- Matlab string
    """
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return "("+str(tree.Emap(-1))+")*d2+("+str(tree.Emap(-2))+")*d3+(A*"+Dprod_str(tree,ThSRKeta_str_matlab)+')'


#=====================================================
#=====================================================
#DEPRECATED FUNCTIONS
#=====================================================
#=====================================================

#=====================================================
def recursiveVectors(p,ind='all'):
#=====================================================
  """ 
    Generate recursive vectors using Albrecht's 'recursion 1' 

    Follows [albrecht1996]_ p. 1718
    Need to extend it for p>10?
  """
  W=[[],[]]
  R=[[],[]]
  R.append(["tau[2]"])
  W.append(["tau[2]"])
  for i in range(3,p):
     #Construct R[i]
     R.append(["tau["+str(i)+"]"])
     for j in range(len(W[i-1])):
        R[i].append("A,"+W[i-1][j])
     #Construct W[i]
     #l=0:
     W.append(R[i][:])
     for l in range(1,i-1): #level 1
        ps=powerString("C",l,trailchar=",")
        for j in range(len(R[i-l])):
          W[i].append(ps+R[i-l][j])
     for l in range(0,i-3): #level 2
        ps=powerString("C",l,trailchar=",")
        for n in range(2,i-l-1):
          m=i-n-l
          if m<=n: #Avoid duplicate conditions
             for Rm in R[m]:
                lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                for Rn in R[n][lowlim:]:
                  W[i].append(ps+Rm+"*"+Rn)
     for l in range(0,i-5): #level 3
        ps=powerString("C",l,trailchar=",")
        for n in range(2,i-l-3):
          for m in range(2,i-l-n-1):
             s=i-m-n-l
             if m<=n and n<=s: #Avoid duplicate conditions
                for Rm in R[m]:
                  lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                  for Rn in R[n][lowlim:]:
                     lowlim2=(n<s and [0] or [R[n].index(Rn)])[0]
                     for Rs in R[s][lowlim2:]:
                        W[i].append(ps+Rm+"*"+Rn+"*"+Rs)
     for l in range(0,i-7): #level 4
        ps=powerString("C",l,trailchar=",")
        for n in range(2,i-l-5):
          for m in range(2,i-l-n-3):
             for s in range(2,i-l-n-m-1):
                t=i-s-m-n-l
                if s<=t and n<=s and m<=n: #Avoid duplicate conditions
                  for Rm in R[m]:
                     lowlim=(m<n and [0] or [R[m].index(Rm)])[0]
                     for Rn in R[n][lowlim:]:
                        lowlim2=(n<s and [0] or [R[n].index(Rn)])[0]
                        for Rs in R[s][lowlim2:]:
                          lowlim3=(s<t and [0] or [R[s].index(Rs)])[0]
                          for Rt in R[t]:
                             W[i].append(ps+Rm+"*"+Rn+"*"+Rs+"*"+Rt)

  if ind=='all': return W[p-1]
  else: return W[p-1][ind]

def py2tex(codestr):
    """Convert a python code string to LaTex"""

    strout=codestr.replace("'","^T")
    strout=strout.replace("*","")
    strout=strout.replace(".^","^")
    strout='$'+strout+'$'
    return strout
