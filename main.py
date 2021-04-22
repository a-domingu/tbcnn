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
from second_neural_network import SecondNeuralNetwork



#####################################
# FUNCTIONS


def first_neural_network(training_dict, feature_size = 5):
    for data in training_dict:
        # Initializing node list, dict list and dict sibling

        #we parse the data of the file into a tree
        tree = file_parser(data)
        # convert its nodes into the Node class we have, and assign their attributes
        ls_nodes, dict_ast_to_Node = node_object_creator(tree)
        ls_nodes = node_position_assign(ls_nodes)
        ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)

        # Initializing vector embeddings
        embed = Embedding(feature_size, ls_nodes, dict_ast_to_Node)
        ls_nodes = embed.node_embedding()

        # Calculate the vector representation for each node
        learning_rate = 0.1
        momentum = 0.01
        vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, feature_size, learning_rate, momentum)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

        training_dict[data] = [ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code]
    return training_dict



def training_dict_set_up(training_path):
    training_set = {}
    for (dirpath, _dirnames, filenames) in os.walk(training_path):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                training_set[filepath] = None
    return training_set

#########################################
# SCRIPT


### Inicializar todos los parametros
alpha = 0.1
epoch = 10
feature_size = 20

# Initialization of layers and SGD
#params, coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer = layer_and_SGD_inizialitation(feature_size, alpha)

### Training set
training_path = "test"

# this is to have all the information of each file in the folder contained in a dictionary
training_dict = training_dict_set_up(training_path)

print(training_dict)

#We now do the first neural network for every file:
training_dict = first_neural_network(training_dict, feature_size)

# Training
secnn = SecondNeuralNetwork()

secnn.train(training_dict, feature_size, epoch)

