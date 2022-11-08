import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class Neuron():
    """
    Core Neuron class based on swc file format
    
    Attributes
    ----------

    name : str
        A string indicating the allocated name of the neuron, by default the file name used to load the data
    
    node_table : np.array
        An array in swc format holding the neuron data

    labels : list
        a list of strings, indicating the order of columns in the node table, and what data is available. Data is as follows:

        Core:
        
            node _id    : Integer ID of specific node 
            label       : The node type - see self.classify_nodes' documentation 
            x           : The x coordinate of the node 
            y           : the y coordinate of the node 
            z           : the z coordinate of the node
            radius      : the radius of the node 
            parent id   : the parent node id indicating the graph edge 

        Can be added:

            distances   : Euclidean distance between parent and child node. 0 if parent is -1(node is root)
    
    summary:  pd.Series
        Summary information for neuron by default including name, number of nodes, end points,branch points, and segments.


    Methods
    -------

    classify_nodes:
            Classify the node types in the label column of the neuron node table

    get_distances:
            Add Euclidean distances between parent and child to the neuron node table

    count_nodes:
            Returns the total number of nodes in the neuron

    count_ends:
            Returns the number of end nodes (leaves) in the neuron

    count_branches:
            Returns the number of branch nodes in the neuron

    count_segments:
            Returns the number of segments which make up the neuron
    
    total_cable_length:
            Returns the total cable length of the neuron


    """
    __slots__ = ['name','node_table','labels','summary']

    def __init__(self,node_table,labels, name = None):

        assert isinstance(labels,list), "Provided column labels are not a list"
        assert isinstance(node_table,np.ndarray), "Provided node table is not an np.array"

        if name is None:
            self.name = None
        else:
            assert isinstance(name,str), "Provided neuron name is not a string"
            self.name = name
        self.node_table = node_table
        self.labels = labels

        self.classify_nodes(overwrite = False)

        df = pd.Series({'Name':self.name,
                        'Nodes':self.count_nodes(),
                        'Ends':self.count_ends(),
                        'Branches':self.count_branches(),
                        'Segments':self.count_segments()})

        self.summary = df

        # add distances
        self.get_distances()


    def classify_nodes(self, overwrite = True):
        """
        Classify nodes in the neuron into root, ends, and branches

        The classifications are encoded as:

            0:'Unlabeled'
            1: 'Soma'
            2: 'Axon'
            3: 'Basal Dendrite'
            4: 'Apical Dendrite'
            5: 'Branch'
            6: 'End'

        Parameters
        ----------

        overwrite   :   bool
            If True (default) will overwrite values in the node table labels column

        Returns
        -------

        None
        """

        # get ind of end nodes
        ends = np.array(list(set(self.node_table[:,self.labels.index('node_id')]) 
                            - set(self.node_table[:,self.labels.index('parent_id')])))
        ends = np.where(np.isin(self.node_table[:,self.labels.index('node_id')],
                                ends, assume_unique = True))

        # get root - this is the inverse of procedure to get ends
        root = np.array(list(set(self.node_table[:,self.labels.index('parent_id')]) 
                            - set(self.node_table[:,self.labels.index('node_id')])))
        root = np.where(np.isin(self.node_table[:,self.labels.index('parent_id')],
                                root, assume_unique = True))

        # get ind of branch nodes

        # we are sorting to get duplicated parents
        branches = np.sort(self.node_table[:,self.labels.index('parent_id')], axis=None)
        branches = branches[:-1][branches[1:] == branches[:-1]]
        # and getting the ind. of the nodes
        branches = np.where(np.isin(self.node_table[:,self.labels.index('node_id')],branches, assume_unique = True))

        # Update labels column

        if overwrite == True:
            self.node_table[:,self.labels.index('label')] = 0

        self.node_table[ends,self.labels.index('label')] = 6
        self.node_table[branches,self.labels.index('label')] = 5
        self.node_table[root,self.labels.index('label')] = -1

    def get_distances(self):
        """
        Add column to the node table with parent to child distances. If the node is the root (it's parent is -1) the distance is 0

        Parameters
        ----------
        
        None

        Returns
        -------

        self    :   Neu_Trees.Neuron
                Appends column to the Neuron node table with the distances. Updates neuron's labels to include distances. Adds total cable length to neuron summary.
        """
        # add distances to node table

        dists = []

        # coords dict - for fast look up 

        coord_dict = {self.node_table[i,0]: self.node_table[i,[2,3,4]] for i in range(self.node_table.shape[0])}

        # loop through each node id
        for i in range(self.node_table.shape[0]):

            # get it's ID
            child = self.node_table[i,0]
            # get the ID of it's parent
            parent = self.node_table[i,6]
            # it it is -1, distance = 0
            if parent == -1:
                dists.append(0)
            # else get coordinate of parent and child
            else:
                child_coord = coord_dict[child]
                parent_coord = coord_dict[parent]
                # get distance and append
                dists.append(np.linalg.norm(child_coord - parent_coord))

        # add column to node table
        self.node_table = np.column_stack((self.node_table,dists))
        # update labels
        self.labels.append('distances')
        # add total cable length to summary
        self.summary['Cable_Length'] = np.sum(dists)

    def count_nodes(self):
        """
        Returns the number of nodes in the neuron
        """
        return self.node_table.shape[0]

    def count_branches(self):
        """
        Returns the number of branch point in the neuron
        """
        return len(np.where(self.node_table[:,self.labels.index('label')] == 5)[0])
    
    def count_ends(self):
        """
        Returns the number of end (leaf) nodes in the neuron
        """
        return len(np.where(self.node_table[:,self.labels.index('label')] == 6)[0])
    
    def count_segments(self):
        """
        Returns the number of segments in the neuron
        """
        return len(np.where(np.isin(self.node_table[:,self.labels.index('label')],[5,6]))[0])

    def total_cable_length(self):
        """
        Returns the total cable length of the neuron.
        """
        if 'distances' not in self.labels:
            self.get_distances()

        return np.sum(self.node_table[:,7])



class NeuronList():
    """
    List object for lists of neurons

    Attributes
    ----------

    neurons:    list
        list of Neu_Trees.Neuron objects

    summary:  pd.DataFrame
        Summary data frame of information of neurons within the Neuron list. By default includes:

            Name:           Neuron Name
            Nodes:          Number of nodes in neuron
            Ends:           Number of end nodes in neuron
            Branches:       Number of branch nodes
            Segments:       Number of segments
            Cable_Length:   Total cable length
     

    Methods
    -------

    classify_nodes:
            Classify the node types in the label column of the neuron node table for all neurons. See 'classify_nodes' documentation for node types. 

    """
      
    __slots__ = ['neurons','summary']

    def __init__(self,Neurons):

        assert isinstance(Neurons,list), "Input is not a list"

        self.neurons = Neurons

        # Create Summary table
        names = [i.name for i in self.neurons]
        nodes = [i.summary.Nodes for i in self.neurons]
        ends = [i.summary.Ends for i in self.neurons]
        branches = [i.summary.Branches for i in self.neurons]
        segments = [i.summary.Segments for i in self.neurons]
        cable = [i.summary.Cable_Length for i in self.neurons]

        df = pd.DataFrame.from_dict({"Name":names,
                                    "Nodes":nodes,
                                    "Ends":ends,
                                    "Branches":branches,
                                    "Segments":segments,
                                    "Cable_Length":cable})

        self.summary = df


    def classify_nodes(self, overwrite = True):
        """
        Classify nodes in the Neuron list into root, ends, and branches

        The classifications are encoded as:

            0:'Unlabeled'
            1: 'Soma'
            2: 'Axon'
            3: 'Basal Dendrite'
            4: 'Apical Dendrite'
            5: 'Branch'
            6: 'End'

        Parameters
        ----------

        overwrite   :   bool
            If True (default) will overwrite values in the node table labels column

        Returns
        -------

        None
        """

        self.neurons = [N.classify_nodes(overwrite = overwrite) for N in tqdm(self.neurons,desc = "Classifying Nodes: ")]


def read_Neuron(path):
    """
    Read a Neuron file format

    Supports: .swc

    Parameters
    ----------

    path    :   str
        File path to neuron file

    Returns
    -------

    Neuron  :   Neu_Trees.Neuron
        Neuron object from file

    """
    if os.path.isfile(path):


        if path.endswith('.swc'):
            N = _read_swc(path)
            return N
        else:
            raise AttributeError('File type not recognised')

    elif os.path.isdir(path):

        N_all = []

        for root, dirs, files in os.walk(path):
            for file in tqdm(files, desc = 'Reading Neurons: '):
                if file.endswith('.swc'):
                    N = _read_swc(os.path.join(root,file))
                    N_all.append(N)

        return NeuronList(N_all)

    else:
        raise TypeError('input is not a file or directory')


def _read_swc(file):
    """
    Read swc data file into Neu_Trees

    Parameters
    ----------

    file    :   str
        File path to neuron .swc file

    Returns
    -------

    Neuron  :   Neu_Trees.Neuron
        Neuron object from .swc file
    """

    # get file name
    name = os.path.splitext(os.path.basename(file))[0]

    # read data and make sure it has 7 columns
    data = np.loadtxt(file)
    assert data.shape[1] == 7, "Expected 7 data columns, got " + str(data.shape[1])

    # get labels (as default swc format)
    labels = ['node_id','label','x','y','z','radius','parent_id']

    return Neuron(data, labels, name)

