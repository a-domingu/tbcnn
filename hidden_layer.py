import torch


class Hidden_layer():

    def __init__(self, vector):
        self.input = vector
        self.n = vector.shape[0]

    def hidden_layer(self):
        parameters = self.initialize_random_vector()
        output = self.get_output(parameters)
        return output

    def initialize_random_vector(self):
        random_vector = torch.rand(self.n)
        return random_vector

    def get_output(self, parameters):
        output = torch.matmul(parameters,self.input)
        return output





