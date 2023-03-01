from .Core import Neuron, Neuron_List
from .Graphs import *
from .Maths import *
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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

    elif isinstance(N,Neuron_List):

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


def upstream_path(N,start,stop = -1, return_length = False):
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
        
        If none of the stop nodes are in the upstream path of the start node, default behavior is to return the path to the root. If this is over-ridden by passing a list to stop not containing -1, NaN is returned in the instance where none of the stop nodes are in the start nodes upstream path

        NOTE: the stop node is NOT included in the returned path

    return_length:  bool
        False by default. If True, path length in the same units as the coordinate space of the original neuron.

    Returns
    -------     

    path:   list | NaN
        list of nodes upstream of the start node to the specified stop node. In the case none of the stop nodes are in the upstream path of the start node, NaN is returned.

    length: float
        The path length of the generated path. only returned if return_path == True

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

    if return_length == False:
        return path
    else:
        dist = upstream_path_length(N,path)
        return path, dist

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

def connected_subset(N,nodes,invert = False):
    """
    Given a list of nodes in N, return the smallest fully connected subset. If invert is True, returns the nodes in N not in the subset

    Parameters
    ----------

    N:          nt.Neuron
        Neuron to collect node subset from

    nodes:      np.array
        np.array of node ids in the neuron we are collecting a subset from.

    invert:     Bool
        False (default) returns the connected subset of nodes connected by the provided nodes, including their first common ancestor. If True, all other nodes in the neuron are returned, EXCLUDING those in the fully connected subset defined by nodes. This will also include the first common ancestor of the set of provided nodes.

    Returns
    -------
    np.array    
        array of node ids in N which make up the smallest connected subset.
    
    """


    A = [set(upstream_path(N,i)) for i in nodes]

    C = list(set.intersection(*A))

    if invert == False:
        # all distances in C
        c = [upstream_path(N,i,return_length=True)[1] for i in C]
        # node in c furthest from root
        c = C[np.where(c == np.max(c))[0][0]]

        S = set.union(*A) - set(C)

        S.add(c)

        # get parent of c
        parent_ind = N.labels.index('parent_id')
        node_ind = N.labels.index('node_id')

        c = N.node_table[np.where(N.node_table[:,node_ind] == c)[0][0],parent_ind]

        S.add(c)

        S = np.array(list(S))

        return S

    else:
        # get all nodes
        node_ind = N.labels.index('node_id')
        nodes = N.node_table[:,node_ind]
        # remove S
        nodes = np.array([i for i in nodes if i not in S])
        # return
        return nodes

def subset_N(N,nodes):
    """
    
    """
    node_ind = N.labels.index('node_id')

    sub_inds = np.where(np.isin(N.node_table[:,node_ind],nodes))

    subset = N.node_table[sub_inds,:][0]

    # update the root
    # get root - this is the inverse of procedure to get ends
    root = np.array(list(set(subset[:,N.labels.index('parent_id')]) 
                        - set(subset[:,N.labels.index('node_id')])))
    root = np.where(np.isin(subset[:,N.labels.index('parent_id')],
                            root, assume_unique = True))

    subset[root[0],N.labels.index('parent_id')] = -1


    return Neuron(subset,N.labels,N.name)

def k_compartments(N,k = 2,outlier_detection = False):
    """
    
    """

    # get ends
    ends = N.get_end_nodes()
    # create igraph object
    g = to_igraph(N,directed = False)

    # step 2 - get dist. matrix and cluster
    g_ends = g.vs.select(lambda vertex: vertex['name'] in ends)
    mat = shortest_distance(g,g_ends,g_ends,weight = 'weight', mode = 'all', flatten = False)

    if outlier_detection == True:
        # convert diagonal to inf
        mat[mat == 0] = np.inf
        # all min distances
        min_dists = np.array([np.min(mat[:,i]) for i in range(mat.shape[0])])
        # closest node id
        min_nodes = np.array([np.where(mat[:,i] == min_dists[i])[0][0] for i in range(len(min_dists))])

        # get nodes in g with think maybe outliers
        t = double_MADs(min_dists, cc= 0.6)

        # get there indicies
        t = np.where(np.isin(min_dists,t))[0]

        # convert to a node id in the original graph
        t = g_ends[t]['name']

        # remove these from ends
        ends = ends[~np.isin(ends,t)]

        # regenerate mat
        g_ends = g.vs.select(lambda vertex: vertex['name'] in ends)
        mat = shortest_distance(g,g_ends,g_ends,weight = 'weight', mode = 'all', flatten = False)

    # perform clustering on path length distance matrix
    cluster = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    cluster.fit_predict(mat)

    # get node ids in original skeleton object\
    labels = np.array([i['name'] for i in g_ends])

    # initialise a list of neurons - for now just collecting their nodes
    Neurons = [labels[np.where(cluster.labels_ == i)[0]] for i in set(cluster.labels_)]

    Neurons = [connected_subset(N,i) for i in Neurons]

    Neurons = [subset_N(N,i) for i in Neurons]

    return Neuron_List(Neurons)



def Eig_align(N,population = False, keep_center = False):
    """
    """            
    # if neuron list
    if isinstance(N, Neuron_List):

        # coordinate indicies
        x_ind = N.neurons[0].labels.index('x')
        y_ind = N.neurons[0].labels.index('y')
        z_ind = N.neurons[0].labels.index('z')
        # if we are rotating globaly
        if population == True:

            # get all coords
            coords=np.vstack([n.get_coords() for n in N.neurons])
            # lets create an array off coords with the neuron names each point belong to do 
            l = [[n.name] * n.count_nodes() for n in N.neurons]
            names = np.array([item for sublist in l for item in sublist])
            original_data = np.c_[names,coords]
            # transpose coords
            coords = coords.T
            # center
            if keep_center == False:
                 # make a note of the centering adjustment
                 centers = np.mean(coords)
            for i in range(coords.shape[0]):
                    coords[i] -= np.mean(coords[i])
            r_coords = snap_to_axis(coords)

            if keep_center == False:
                 for i in range(coords.shape[0]):
                      r_coords[i] += centers[i]
            
            # transpose r_coords
            r_coords = r_coords.T
            # overwrite the original data with the new coordinates
            original_data[:,1:4] = r_coords
            # update coordinates in Neurons with new data

            for i in range(len(N.neurons)):
                n = N.neurons[i]
                current = original_data[np.where(original_data[:,0]==n.name)[0],1:4]
                n.node_table[:,[x_ind,y_ind,z_ind]] = current
                N.neurons[i] = n

        # if we are rotating each individual Neuron
        else:
             for n in range(len(N.neurons)):
                n = N.neurons[i]
                # get coords
                coords = n.get_coords().T
                # center - note keep center
                if keep_center == False:
                    centers = np.mean(coords)
                for i in range(coords.shape[0]):
                     coords[i] -= np.mean(coords[i])
                # rotate
                r_coords = snap_to_axis(coords)
                # undo centering if we need to 
                if keep_center == False:
                     for i in range(coords.shape[0]):
                          r_coords[i] += centers[i]
                # update coordinates
                n.node_table[:,[x_ind,y_ind,z_ind]] = r_coords.T
                N.neurons[i] = n
    elif isinstance(N, Neuron):
        # inds
        x_ind = N.labels.index('x')
        y_ind = N.labels.index('y')
        z_ind = N.labels.index('z')
        #  get coords
        coords = N.get_coords().T
        # center - note keep center
        if keep_center == False:
            centers = np.mean(coords)
        for i in range(coords.shape[0]):
            coords[i] -= np.mean(coords[i])
        # rotate
        r_coords = snap_to_axis(coords)
        # undo centering if we need to 
        if keep_center == False:
            for i in range(coords.shape[0]):
                r_coords[i] += centers[i]
        N.node_table[:[x_ind,y_ind,z_ind]] = r_coords.T

    return N

def Tree_height(N):
    """
    
    """
    # get ends
    ends = np.unique(N.get_end_nodes())
    # get branches
    branches = np.unique(N.get_branch_nodes())
    # initialise heights
    heights = []
    # for each end:
    for e in ends:
        # get path to root
        path = np.unique(nt.upstream_path(N,start = e))
        # get number of branches in path
        heights.append(len(np.intersect1d(path,branches)))

    # return the maximum value
    return np.max(heights)
