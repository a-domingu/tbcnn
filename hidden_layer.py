import torch


class Hidden_layer():
    
    def __init__(self, output_size = 4):
        self.ls = []
        self.input = []
        self.w = None
        self.b = None
        self.Nc = output_size
        # The size of n is based on the dynamic pooling method.
        # In one-way pooling the size of n is equal to the output_size / feature_detectors
        # In three-way pooling the size of n is equal to 3
        self.n = 3

    def initialize_parameters(self):
        self.w = torch.randn(self.n, requires_grad = True)
        self.b = torch.randn(1, requires_grad = True)

        return self.w, self.b

    def hidden_layer(self, ls_nodes, vector):
        # Initialize the node list and the vector
        self.ls = ls_nodes
        self.input = vector

        output = self.get_output()
        return output

    def get_output(self):
        output = torch.matmul(self.w,self.input) + self.b
        return output