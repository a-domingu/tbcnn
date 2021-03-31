import ast
from relu import relu
import numpy as np

class Node():
    def __init__(self, node, parent = None):
        self.node = node
        self.children = self.get_children()
        self.parent = parent
        self.type = self.node.__class__.__name__
        self.vector = self.get_vector

    def __str__(self):
        return self.type


    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            nodeChild = Node(child)
            ls.append(nodeChild)
        return ls

    def get_vector(self):
        return self.vector

    def set_vector(self, vector):
        self.vector = vector

    def update_vector(self, w, b):
        '''
        This function's purpose is to get the update vector representations of each node, 
        following the process of the bug detection article
        '''
        # asegurarnos que self.vector no sea una lista de python, sino un np.array
        z = np.dot(w, self.vector) + b
        self.new_vector = relu(z)
        return self.new_vector