import numpy as np
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

class Convolutional_layer_algorithm():
    """
    """

    def __init__(self, ls_nodes, dict_ast_to_Node, features_size):
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node
        self.features_size = features_size
        self.w_t = None
        self.w_r = None
        self.w_l = None
        self.b_conv = None
        self.y = None

    def convolutional_layer(self):
        # Parameters initialization.
        # The matrices w_t, w_r, w_l and the vector b_conv must be initialized randomly.
        matrices = MatrixGenerator(self.ls, self.features_size)
        self.w_t = matrices.w
        self.w_r = matrices.w
        self.w_l = matrices.w
        self.b_conv = matrices.b

        # self.y is the output of the convolutional layer.
        self.y = self.calculate_y()

        return self.y

    def calculate_y(self):
        #Initialize the sum
        sum = torch.zeros(self.features_size) 

        # Iterate along depth window, not self.ls. #TODO

        for node in self.ls:

            # Adapt parameters to calculate the weight matrix to each node.
            #TODO
            # Parameters used to calculate the weight matrix for each node
            d_i = None #TODO   # d_i : the depth of the node i in the sliding window
            d   = None #TODO   # d: the depth of the window
            p_i = None #TODO   # p_i the position of the node
            n   = len(node.parent.children)   # n: the total number of p's siblings

            #d_i -> si un nodo tiene cuatro "hermanos" (por ejemplo), seria 1 para el primero,
            # dos para el segundo, etc...
            #d -> distancia del nodo a la raíz
            #n -> numero de hermanos

            # The weighted matrix for each node is a linear combination of matrices w_t, w_l and w_r
            weighted_matrix = self.weight_matrix_update(d_i, d, p_i, n)

            sum = sum + weighted_matrix*node.combined_vector

        # When all the "weighted vectors" are added, we add on the b_conv.
        argument = sum + self.b_conv

        # We used relu as the activation function in TBCNN mainly because we hope to 
        # encode features to a same semantic space during coding.
        self.y = F.relu(argument)

    def weight_matrix_update(self, d_i, d, p_i, n):
        # The matrices coefficients are computed according to the relative position of 
        # a node in the sliding window.

        n_t = (d_i - 1)/(d-1)       # Coefficient associated to w_t
        n_r = (1-n_t)*(p_i-1)/(n-1) # Coefficient associated to w_r
        n_l = (1-n_t)(1-n_r)        # Coefficient associated to w_l 

        top_matrix = n_t*self.w_t
        left_matrix = n_l* self.w_l
        right_matrix = n_r*self.w_r
        return (top_matrix + left_matrix + right_matrix) 