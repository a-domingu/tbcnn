import numpy as np




class MatrixGenerator():
    ''' This class' purpose is to receive the list of nodes of a project as input, 
    and using their vector representations to generate a matrix using the idea behind 
    TBCNN, so that we can have the necessary input for a CNN.'''

    def __init__(self, ls_nodes, n):
        self.ls_nodes = ls_nodes
        self.w = self.initalize_random_weight_matrix(n)
        self.b = self.initalize_random_bias_vector(n)


    def initalize_random_weight_matrix(self, n):
        # 'n' refers to the number of rows we want our matrix to have
        initial_node = self.ls_nodes[0]
        m = len(initial_node.vector)
        weight_matrix = np.random.rand(n,m)
        return weight_matrix

    def initalize_random_bias_vector(self, n):
        bias = np.random.rand(n)
        return bias

    

    









