import ast
from relu import relu
import numpy as np
import torch as torch


class Node():
    '''
    For each node we store its parent and children nodes, as well as, its node type and its vector 
    representation
    '''
    def __init__(self, node, parent = None):
        self.node = node
        self.children = self.get_children()
        self.parent = parent
        self.type = self.node.__class__.__name__
        self.vector = []
        self.combined_vector = []

    def __str__(self):
        return self.type

    #Returns the children nodes of each node
    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            #nodeChild = Node(child, self)
            ls.append(child)
        return ls

    #Assigns the vector embedding to each node
    def set_vector(self, vector):
        if type(vector) == torch.Tensor:
            self.vector = vector
        else:
            self.vector = torch.tensor(vector, requires_grad = True)
    
    def set_combined_vector(self, vector):
        self.combined_vector = vector

    def update_vector(self, w, b):
        '''
        This function's purpose is to get the update vector representations of each node, 
        following the process of the bug detection article
        '''
        # asegurarnos que self.vector no sea una lista de python, sino un np.array
        z = torch.matmul(w*self.vector) + b
        self.new_vector = relu(z)
        return self.new_vector
        