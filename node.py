import ast




class Node():
    def __init__(self, node, parent = None):
        self.node = node
        self.children = self.get_children()
        self.parent = parent
        self.type = self.node.__class__.__name__

    def __str__(self):
        return self.type



    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            ls.append(child)
        return ls


    def get_vector(self):
        # TODO implementar lo de Esther aqui
        self.vector = ['vector de prueba', self.node] #cambiar esto por lo que corresponda al meter lo que ha hecho Esther aqui
        return self.vector


