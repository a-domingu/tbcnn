import torch







class Pooling_layer():

    '''
    This class will receive a list of nodes (of 'Node' type), from which we'll take their node.y vector,
    and apply the max pool function. This function will simply return the maximum element of node.y 
    (infinity norm), and we'll save it as an atribute of each node
    '''

    def __init__(self, ls_nodes):
        self.ls = ls_nodes

    '''
    def pooling_layer(self):
        for node in self.ls:
            y = node.y
            pool = torch.max(y)
            node.set_pool(pool)
    '''
    def pooling_layer(self):
        ls_tensors = self.create_ls_tensors(self.ls)
        matrix = self.create_matrix(ls_tensors)
        print('tree_tensor: \n', tree_tensor)
        pooled_tensor = self.one_way_pooling(tree_tensor)
        print('pooled_tensor:\n', pooled_tensor)

    def create_ls_tensors(self, ls_nodes):
        ls_tensors = []
        for node in ls_nodes:
            ls_tensors.append(node.y)
        return ls_tensors
    
    def create_matrix(self, ls):
        matrix = torch.stack(ls)
        return matrix
    
    def one_way_pooling(self, tensor):
        pool_tensor = torch.max(tensor, dim = 0)
        return pool_tensor
        






