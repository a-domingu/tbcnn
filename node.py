import ast

from gensim.models import Word2Vec

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