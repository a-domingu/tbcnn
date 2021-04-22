import sys
import os
import gensim
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

from node_object_creator import *
from embeddings import Embedding
from node import Node
from matrix_generator import MatrixGenerator
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer
from get_targets import GetTargets


class SecondNeuralNetwork():

    def __init__(self, n = 20, m = 4):
        self.vector_size = n
        self.feature_size = m
        #parameters
        self.w_comb1 = torch.diag(torch.randn(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_comb2 = torch.diag(torch.randn(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_t = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.w_r = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.w_l = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.b_conv = torch.randn(self.feature_size, requires_grad = True)
        # usar linea de abajo cuando sea con one_way_pooling
        #self.w_hidden = torch.randn(self.feature_size, requires_grad = True)
        # y la de abajo para dynamic_pooling
        self.w_hidden = torch.randn(3, requires_grad = True)
        self.b_hidden = torch.randn(1, requires_grad = True)
        # layers
        self.cod = Coding_layer(self.vector_size, self.w_comb1, self.w_comb2)
        self.conv = Convolutional_layer(self.vector_size, self.w_t, self.w_r, self.w_l, self.b_conv, features_size=self.feature_size)
        self.dynamic = Dynamic_pooling_layer()
        self.max_pool = Max_pooling_layer()
        self.hidden = Hidden_layer(self.w_hidden, self.b_hidden)


    

    def train(self, targets, training_dict, total_epochs):
        """Create the training loop"""
        # Construct the optimizer
        params = [self.w_comb1, self.w_comb2, self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]
        optimizer = torch.optim.SGD(params, lr = 0.1)
        criterion = nn.BCELoss()
        print(targets)

        for epoch in range(total_epochs):
            # Time
            start = time()
            outputs = self.forward(training_dict)

            loss = criterion(outputs, targets)
            
            # zero the parameter gradients
            print('outputs: \n', outputs)
            #print('Matrix w_r_conv: \n', params[4])

            optimizer.zero_grad()

            # Calculates the derivative
            loss.backward(retain_graph = True)

            # Update parameters
            optimizer.step() #w_r = w_r - lr * w_r.grad

            #Time
            end = time()

            print('Epoch ', epoch, ', Time: ', end-start, 'Loss: ', loss)


    def forward(self, training_dict):
        outputs = []
        softmax = nn.Sigmoid()
        for filepath in training_dict.keys():
            data = filepath
            
            ## forward (layers calculations)
            output = self.layers(training_dict[data])

            # output append
            if outputs == []:
                outputs = softmax(output)
            else:
                outputs = torch.cat((outputs, softmax(output)), 0)

        return outputs


    def layers(self, vector_representation_params):
        ls_nodes = vector_representation_params[0]
        dict_ast_to_Node = vector_representation_params[1]
        dict_sibling = vector_representation_params[2]
        w_l_code = vector_representation_params[3]
        w_r_code = vector_representation_params[4]
        b_code = vector_representation_params[5]
        ls_nodes = self.cod.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code)
        ls_nodes = self.conv.convolutional_layer(ls_nodes, dict_ast_to_Node)
        self.max_pool.max_pooling(ls_nodes)
        vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        #vector = pooling_layer.pooling_layer(ls_nodes)
        output = self.hidden.hidden_layer(vector)

        return output

