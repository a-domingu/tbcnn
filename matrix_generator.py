import numpy as np




class MatrixGenerator():
    ''' This class' purpose is to receive the list of nodes of a project as input, 
    and using their vector representations to generate a matrix using the idea behind 
    TBCNN, so that we can have the necessary input for a CNN.'''

    def __init__(self, ls_nodes):
        self.ls_nodes = ls_nodes

    def initalize_random_parameters(self, n):
        # 'n' refers to the number of rows we want our matrix to have
        initial_node = self.ls_nodes[0]
        m = len(initial_node.vector)
        weight_matrix = np.random.rand(n,m)
        bias = np.random.rand(n)
        return weight_matrix, bias

    









