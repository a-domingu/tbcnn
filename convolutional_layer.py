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

    def __init__(self, ls_nodes, dict_ast_to_Node, features_size, kernel_depth = 2):
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node
        self.features_size = features_size
        self.w_t = None
        self.w_r = None
        self.w_l = None
        self.b_conv = None
        self.y = None
        self.Nc = None
        self.kernel_depth = kernel_depth

    def convolutional_layer(self):
        # We calculate the number of feature detectors (N_c). Feature detectors == number of nodes - number of leaf nodes
        self.Nc = 0
        for node in self.ls:
            if node.children:
                self.Nc +=1

        # Parameters initialization.
        # The matrices w_t, w_r, w_l and the vector b_conv must be initialized randomly.
        matrices = MatrixGenerator(self.ls, self.Nc)
        self.w_t = matrices.w
        self.w_r = matrices.w
        self.w_l = matrices.w
        self.b_conv = matrices.b

        # self.y is the output of the convolutional layer.
        self.y = self.calculate_y()

        return self.y

    def calculate_y(self):

        for node in self.ls:
            if node.children:
                ''' We are going to create the sliding window. Taking as reference the book,
                we are going to set the kernel depth of our windows as 2. We consider into the window
                the node and its children.

                Question for ourselves: if we decide to increase the kernel depth to 3, should be
                appropiate to take node: its children and its grand-children or node, parent and children?
                '''
                # We create the sliding window with kernel depth = 2.
                window_nodes = [node]
                for child in node.children:
                    window_nodes.append(self.dict_ast_to_Node[child])

                sum = torch.zeros(self.Nc)

                d = node.depth + self.kernel_depth - 1
                for item in window_nodes:
                    # Parameters used to calculate the weight matrix for each node
                    d_i = d - item.depth + 1
                    p_i = item.position
                    n = item.siblings
                    # The weighted matrix for each node is a linear combination of matrices w_t, w_l and w_r
                    weighted_matrix = self.weight_matrix_update(d_i, d, p_i, n)

                    sum = sum + weighted_matrix*item.combined_vector

                # When all the "weighted vectors" are added, we add on the b_conv.
                argument = sum + self.b_conv

                # We used relu as the activation function in TBCNN mainly because we hope to 
                # encode features to a same semantic space during coding.
                node.set_y(F.relu(argument))

            else:
                node.set_y(torch.zeros(self.Nc))

    def weight_matrix_update(self, d_i, d, p_i, n):
        # The matrices coefficients are computed according to the relative position of 
        # a node in the sliding window.

        n_t = (d_i - 1)/(d-1)       # Coefficient associated to w_t
        n_r = (1-n_t)*(p_i-1)/(n-1) # Coefficient associated to w_r
        n_l = (1-n_t)*(1-n_r)        # Coefficient associated to w_l 

        top_matrix = n_t*self.w_t
        left_matrix = n_l* self.w_l
        right_matrix = n_r*self.w_r
        return (top_matrix + left_matrix + right_matrix) 