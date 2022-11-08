from .Core import Neuron, NeuronList
from scipy.spatial import ConvexHull

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

