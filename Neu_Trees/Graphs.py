from .Core import Neuron
from .Maths import *
import igraph as ig
import numpy as np
from scipy.spatial.distance import squareform


def to_igraph(N,directed = False):
    """
    Return original neuron as an igraph.Graph object.

    Parameters
    ----------

    N:          Neu_Trees.Neuron
        Neu_Trees.Neuron object to convert to an igraph graph object

    directed:   Bool
        If True, returns a directed graph, otherwise returns an undirected graph (default)

    Returns
    -------

    g:          igraph.Graph
        directed or undirected igraph tree graph representing the neuron.

    """

    # initialise empty graph
    if directed == True:
        g = ig.Graph(directed = True)
    else:
        g = ig.Graph()

    # vertex properties
    vertex_properties = {'name':N.node_table[:,N.labels.index('node_id')],
                        'x':N.node_table[:,N.labels.index('x')],
                        'y':N.node_table[:,N.labels.index('y')],
                        'z':N.node_table[:,N.labels.index('z')]}
    g.add_vertices(N.node_table.shape[0],vertex_properties)

    # get edges and add
    edges = N.node_table[:,[N.labels.index('node_id'),N.labels.index('parent_id')]].astype(int)
    edges = [tuple(i) for i in edges if i[1] != -1]
    new_ids = list(range(N.node_table.shape[0]))
    old_ids = list(N.node_table[:,N.labels.index('node_id')])
    mapping = {old_ids[i]:new_ids[i] for i in range(len(new_ids))}
    edges = [(mapping[i[0]], mapping[i[1]]) for i in edges]

    g.add_edges(edges)

    if 'distances' in N.labels:
        # get edge lengths and add as property to graph
        root_ind = np.where(N.node_table[:,N.labels.index('parent_id')] == -1)[0][0]
        # distances array
        dists = N.node_table[~np.isin(np.arange(len(N.node_table)), root_ind),N.labels.index('distances')]
        g.es['weight'] = dists
    
    return g


def shortest_distance(g,source = None, target = None, weight = None, mode = 'all', force_symmetry = True, flatten = True):
    """
    Returns path length distance matrix between nodes

    Parameters
    ----------
    g:              igraph.Graph
                igraph graph object to use

    source:         igraph.vertex | list | igraph.vs | None
                Node(s) to use as source. if None, all nodes will be used

    target:         igraph.vertex | list | igraph.vs | None
                Node(s) to use as target. if None, all nodes will be used


    weight:         None | stg
                If none, path length is based on number of edges in path. If string is passed, must be the key name which denotes weight in the original graph object, g

    mode:           str
                Accepted inputs: 'all', 'in', 'out'. see g.shortest_paths documentation for details. 'all' by default

    force_symmetry:  bool
                If True, returned matrix will have symmetry imposed on it by extracting the upper triangle of the matrix and adding it to it's transpose.


    flatten:        bool
                If True, array will be forced to by symmetric and then flattened


    Returns
    -------

    np.ndarray      np.array    
                Array of shortest path lengths between source and target node(s)
    
    """

    if not isinstance(g,ig.Graph):
        raise TypeError('give g is not an igraph.Graph object')

    mat = np.array(g.distances(source = source,target = target, weights='weight', mode = 'all'))

    if force_symmetry == True:

        if check_symmetric(mat) == False:
            mat = np.triu(mat)
            mat = mat + mat.T

    if flatten == True:
        if check_symmetric(mat) == False:
            mat = np.triu(mat)
            mat = mat + mat.T
        mat = squareform(mat)

    return mat



