from .Core import Neuron, NeuronList
from scipy.spatial import ConvexHull
import numpy as np

def hull_metrics(N, update = True, ret = False):
    """
    Returns the surface area and volume of a convex hull fitted to the neurons points. Convex hull is fitted using 'scipy.spatial.ConvexHull'

    Parameters
    ----------

    Neuron:         Neu_Trees.Neuron | Neu_Trees.NeuronList
            A single neuron, or neuron list, to fit convex hull(s) to and extract metrics

    Returns
    -------

    Surface area:   int | list
            The surface area of the convex hull surrounding the neurons points

    Volume:         int | list
            The volume of the convex hull surrounding the neurons points

    """

    if isinstance(N,Neuron):

        coords = N.node_table[:,[2,3,4]]
        hull = ConvexHull(coords)

        if update == True:
            N.summary['Area'] = hull.area
            N.summary['Volume'] = hull.volume

        
        if ret == True:
            return hull.area, hull.volume
        else:
            pass

    elif isinstance(N,NeuronList):

        area = []
        volume = []
        for n in N.neurons:
            coords = n.node_table[:,[2,3,4]]
            hull = ConvexHull(coords)
            area.append(hull.area)
            volume.append(hull.volume)

        if update == True:
            N.summary['Area'] = area
            N.summary['Volume'] = volume

        if ret == True:
            return area,volume
        else:
            pass

    else:
        raise TypeError("input type is not Neuron or NeuronList")


def upstream_path(N,start,stop = -1):
    """
    Given a starting node, return the upstream path - going from child to parent - for that node to the root (default) or specified list of upstream nodes. 

    If the upstream path of a given start node is not in the list of given stop nodes, the path will default to returning the path of the given start node to the root unless -1 is removed from the stop list. In such a case NaN will be returned as the start node has no upstream path to (any of) the given stop nodes. 

    Parameters
    ----------

    N:      Neu_Trees.Neuron
        The Neuron object from which upstream paths are being collected

    start:  int | float
        The start node from which to generate the upstream path

    stop:   int | list
        node(s) at which to stop path collection at. By default, the path to the root of the tree is returned. 

        If a list of stop nodes is given, the path will stop at the first node encountered upstream within the stop list.
        
        If none of the stop nodes are in the upstream path of the start node, default behaviour is to return the path to the root. If this is over-riden by passing a list to stop not containing -1, NaN is returned in the instance where none of the stop nodes are in the start nodes upstream path

    Returns
    -------     

    path:   list | NaN
        list of nodes upstream of the start node to the specified stop node. In the case none of the stop nodes are in the upstream path of the start node, NaN is returned.

    """

    if isinstance(stop, int):
        stop = [stop]

    # for indexing...
    node_ind = N.labels.index('node_id')
    parent_ind = N.labels.index('parent_id')

    # initialise path and starting node
    path = []
    current = start

    while current not in stop:

        if current == -1:
            print("None of the given stop nodes are in the upstream path of the start node. return NaN")
            return np.nan

        # add the current node to the path
        path.append(current)

        # find the current node
        child = np.where(N.node_table[:, node_ind] == current)[0][0]
        
        # get its parent
        parent = N.node_table[child,parent_ind]

        # update current
        current = parent

    return path

def upstream_path_length(N,path):
    """
    Given a path of neurons in the tree graph, return the total path length. 

    Note - if a non-existent path is given, a path length will still be returned, but ill be incorrect.

    Parameters
    ----------

    N:          Neu_Trees.Neuron
        Neuron object where the upstream path originated from 

    path:       list
        list representing the nodes in the upstream path for which to calculate the distance

    Returns
    -------

    distance:   float
        The path length of the path, in the same units as the original Neuron coordinate system.
    
    """
    # upstream path length
    if 'distance' not in N.labels:
        N.get_distances()

    # inds for indexing...
    node_ind = N.labels.index('node_id')
    dist_ind = N.labels.index('distances')

    # get the ind. of nodes in the path 
    path_inds = np.where(np.isin(N.node_table[:,node_ind],path))[0]

    # distances
    return sum(N.node_table[path_inds,dist_ind])






