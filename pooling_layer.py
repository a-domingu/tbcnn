import torch

class Max_pooling_layer():

    '''
    This class will receive a list of nodes (of 'Node' type), from which we'll take their node.y vector,
    and apply the max pool function. This function will simply return the maximum element of node.y 
    (infinity norm), and we'll save it as an atribute of each node
    '''

    def __init__(self, ls_nodes):
        self.ls = ls_nodes

    def max_pooling(self):
        for node in self.ls:
            y = node.y
            pool = torch.max(y)
            node.set_pool(pool)


class Dynamic_pooling_layer():

    def __init__(self, ls_nodes, dict_sibling, nb_slots = 3):
        self.ls = ls_nodes
        self.dict_sibling = dict_sibling
        self.nb_slots = nb_slots 
        # Number of nodes in each slot
        self.nodes_per_slot = int(len(ls_nodes)/nb_slots) 
        self.ls_top = []
        self.ls_left = []
        self.ls_right = []
        self.pooling_vector = None

    def three_way_pooling(self):
        top_depth = self.top_slot()
        self.left_right_slot(top_depth)
        top_max = max(self.ls_top)
        left_max = max(self.ls_left)
        right_max = max(self.ls_right)
        self.pooling_vector = torch.stack((top_max, left_max, right_max), dim=0)

        return self.pooling_vector


    def top_slot(self):
        for depth in self.dict_sibling.keys():
            vector_depth = self.dict_sibling[depth]
            for nodo in vector_depth:
                self.ls_top.append(nodo.pool)
            if len(self.ls_top) >= self.nodes_per_slot:
                top_depth = depth
                break
        return top_depth
    

    def left_right_slot(self, top_depth):
        for depth in self.dict_sibling.keys():
            if depth > top_depth:
                vector_depth = self.dict_sibling[depth]
                division_criteria = int(len(vector_depth)/2)
                for nodo in vector_depth:
                    if nodo.position < division_criteria:
                        self.ls_left.append(nodo.pool)
                    else:
                        self.ls_right.append(nodo.pool)







