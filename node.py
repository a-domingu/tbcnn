import ast
from relu import relu
import numpy as np

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
        self.vector = self.get_vector
        self.counter = 0

    def __str__(self):
        return self.type

    #Returns the children nodes of each node
    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            nodeChild = Node(child, self)
            ls.append(nodeChild)
        return ls

    #Returns the vector embedding of each node
    def get_vector(self):
        return self.vector

    #Assigns the vector embedding to each node
    def set_vector(self, vector):
        self.vector = np.array(vector)

    def update_vector(self, w, b):
        '''
        This function's purpose is to get the update vector representations of each node, 
        following the process of the bug detection article
        '''
        # asegurarnos que self.vector no sea una lista de python, sino un np.array
        z = np.dot(w, self.vector) + b
        self.new_vector = relu(z)
        return self.new_vector

    def get_l(self):
        '''
        This function's output is the number of leaf nodes under each node
        '''
        leaves_under_node = 0
        if len(self.children) == 0:
            return leaves_under_node
        else:
            leaves_under_node = self.calculate_l(self.node, leaves_under_node)
        return leaves_under_node


    def calculate_l(self, node, leaves_under_node):
        for child in node.children:
            if len(child.children) == 0:
                leaves_under_node += 1
            else:
                leaves_under_node = calculate_l(child, leaves_under_node)
        return leaves_under_node
        