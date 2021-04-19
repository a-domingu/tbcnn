import torch

from matrix_generator import MatrixGenerator

class Hidden_layer():

    def __init__(self, ls_nodes, vector):
        self.ls = ls_nodes
        self.input = vector
        self.w = None
        self.b = None
        self.n = vector.shape[0]

    def hidden_layer(self):
        self.initialize_random_parameters()
        output = self.get_output()
        return output, self.w, self.b

    def initialize_random_parameters(self):
        matrices = MatrixGenerator(self.ls, self.n)
        self.w = matrices.w
        self.b = matrices.b

    def get_output(self):
        output = torch.matmul(self.w,self.input) + self.b
        return output





