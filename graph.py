import networkx as nx
import numpy as np

def longest_paths(G):
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

def rk_dependency_graph(rkm):
    """Dependency graph of Runge-Kutta method stages"""
    s = len(rkm)
    K = np.zeros((s+1,s+1))
    K[:,:-1] = np.vstack((rkm.A,rkm.b))
    G = nx.from_numpy_matrix(np.abs(K.T)>0,nx.DiGraph())
    return G

def plot_dependency_graph(rkm):
    G = rk_dependency_graph(rkm)
    lp = longest_paths(G)

    pos = {}
    for node in G.nodes_iter():
        pos[node] = np.array([node,lp[node]])

    nx.draw_networkx(G,pos)
