import sys
import os
import shutil
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
from validation_neural_network import Validation_neural_network


#####################################
# SCRIPT
def main():
    ### Inicializar todos los parametros
    # First neural network parameters
    vector_size = 20
    learning_rate = 0.1
    momentum = 0.01
    # Second neural network parameters
    learning_rate2 = 0.1
    feature_size = 4
    epoch = 4
    pooling = 'one-way pooling'

    ### Creation of the training set and validation set
    path = 'test'
    training_dict, validation_dict = training_and_validation_sets_creation(path) 
    print('Training dict: ', training_dict)
    print('Validation dict: ', validation_dict)

    
    ### Training set
    #training_path = '..\\training'
    # this is to have all the information of each file in the folder contained in a dictionary
    #training_dict = training_dict_set_up(training_path)
    # this is the tensor with all target values
    targets = target_tensor_set_up(path, training_dict)

    print('Target set: ', targets)
    
    # We now do the first neural network for every file:
    training_dict = first_neural_network(training_dict, vector_size, learning_rate, momentum)

    # Training
    secnn = SecondNeuralNetwork(vector_size, feature_size, pooling)
    secnn.train(targets, training_dict, epoch, learning_rate2)

    # Validation
    #validation_path = '..\\validation'
    val = Validation_neural_network(vector_size, feature_size, pooling)
    val.validation(path, validation_dict)


#####################################
# FUNCTIONS

def training_and_validation_sets_creation(path):
    # we create the training set and the validation set
    training_set = {}
    validation_set = {}
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, dirnames, filenames) in os.walk(path):
        for folder in dirnames:
            # we list all files of each folder
            folder_path = os.path.join(dirpath, folder)
            list_dir = os.listdir(folder_path)
            # Having a list with only .py files
            list_files_py = [file for file in list_dir if file.endswith('.py')]
            print(list_files_py)
            # we choose randomly 70% of this files
            # Number of files in the training set
            N = int(len(list_files_py)*0.7)
            i=1
            while list_files_py:
                file = random.choice(list_files_py)
                list_files_py.remove(file)
                if i <= N:
                    filepath = os.path.join(folder_path, file)
                    training_set[filepath] = None
                else:
                    filepath = os.path.join(folder_path, file)
                    validation_set[filepath] = None
                i += 1
    return training_set, validation_set


def target_tensor_set_up(path, training_dict):
    # Target dict initialization
    target = GetTargets(path)
    targets_dict = target.df_iterator()
    print(targets_dict)
    targets = []
    for filepath in training_dict.keys():
        # Targets' tensor creation
        split_filepath = os.path.split(filepath)
        #print('Splitting the filepath: ', split_filepath)
        filepath_target = 'label_' + split_filepath[1] + '.csv'
        #print('Modified file: ', filepath_target)
        search_target = os.path.join(split_filepath[0], filepath_target)
        #print('Final path: ', search_target)
        if search_target in targets_dict.keys():
            if targets == []:
                targets = targets_dict[search_target]
            else:
                targets = torch.cat((targets, targets_dict[search_target]), 0)
    print("target tensor:", targets)
    return targets


def first_neural_network(training_dict, vector_size = 20, learning_rate = 0.1, momentum = 0.01):
    for data in training_dict:
        # Initializing node list, dict list and dict sibling

        # we parse the data of the file into a tree
        tree = file_parser(data)
        # convert its nodes into the Node class we have, and assign their attributes
        ls_nodes, dict_ast_to_Node = node_object_creator(tree)
        ls_nodes = node_position_assign(ls_nodes)
        ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)

        # Initializing vector embeddings
        embed = Embedding(vector_size, ls_nodes, dict_ast_to_Node)
        ls_nodes = embed.node_embedding()

        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, vector_size, learning_rate, momentum)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

        training_dict[data] = [ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code]
        print("end vector representation of file:", data)
    return training_dict


'''
def training_dict_set_up(training_path):
    training_set = {}
    for (dirpath, _dirnames, filenames) in os.walk(training_path):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                training_set[filepath] = None
    return training_set
'''

'''
def directories_creation(path):
    # Check if a directory exists; otherwise create it
    if os.path.isdir(path):
        # we list all files in the directory
        list_dir = os.listdir(path)
        # we delete all files in the directory
        for filename in list_dir:
            file_path = os.path.join(path, filename)
            #deleting file
            os.unlink(file_path)
    else:
        os.makedirs(path)
'''
########################################


if __name__ == '__main__':
    main()

