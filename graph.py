"""
Routines for plotting Runge-Kutta method stage dependency graphs.

The stage dependency graph shows the order in which stages must be computed.
For traditional methods, this is not very interesting, but for special classes
of methods this reveals the potential for parallelism.

**Examples**::

    >>> from nodepy import rk
    >>> ex4 = rk.extrap(4)
    >>> plot_dependency_graph(ex4)
"""
def plot_dependency_graph(rkm,remove_edges=False):
    """Plot the stage dependency graph of a Runge-Kutta method.
    
    **Inputs**:

        - rkm : A RungeKuttaMethod
    """
    import networkx as nx
    import numpy as np
    G = rk_dependency_graph(rkm)
    lp = _longest_paths(G)

    hpos = [0]*(max(lp)+1)
    pos = {}
    for node in G.nodes_iter():
        hpos[lp[node]]+=1
        pos[node] = np.array([hpos[lp[node]],lp[node]])

    if remove_edges:
       _remove_extra_edges(G)

    nx.draw_networkx(G,pos)

def rk_dependency_graph(rkm):
    """Construct the dependency graph of the stages of a Runge-Kutta method."""
    import networkx as nx
    import numpy as np
    s = len(rkm)
    K = np.zeros((s+1,s+1))
    K[:,:-1] = np.vstack((rkm.A,rkm.b))
    G = nx.from_numpy_matrix(np.abs(K.T)>0,nx.DiGraph())
    return G

def _longest_paths(G):
    """Return a dict of nodes with values corresponding to longest paths from node"""
    lp = {}
    lp_old = lp.copy()
    for node in G.nodes_iter():
        lp[node] = 0

    while lp != lp_old:
        lp_old = lp.copy()
        for edge in G.edges_iter():
            lp[edge[1]] = max(lp[edge[1]],lp[edge[0]]+1)

    return lp

def _remove_extra_edges(G):
    """Remove edges that aren't needed in establishing ordering."""
    lp = _longest_paths(G)
    to_remove = []
    for edge in G.edges_iter():
        if lp[edge[0]]<(lp[edge[1]]-1):
            to_remove.append(edge)
    for edge in to_remove:
        G.remove_edge(edge[0],edge[1])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
