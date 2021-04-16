import torch

class Pooling_layer():

    '''
    This class will receive a list of nodes (of 'Node' type), from which we'll take their node.y vector,
    and apply the max pool function. This function will simply return the maximum element of node.y 
    (infinity norm), and we'll save it as an atribute of each node
    '''

    def __init__(self, ls_nodes):
        self.ls = ls_nodes

    def pooling_layer(self):
        for node in self.ls:
            y = node.y
            pool = torch.max(y)
            node.set_pool(pool)





