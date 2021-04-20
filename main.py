import sys
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
from vector_representation import Vector_representation_algorithm
from coding_layer import Coding_layer_algorithm
from convolutional_layer import Convolutional_layer_algorithm
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer


#####################################
# FUNCTIONS

def train(coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, folder, feature_size, optimizer, epoch):
    """Create the training loop"""
    criterion = nn.BCELoss()
    softmax = nn.Sigmoid()
    for step in range(epoch):
        # Time
        start = time()
        # Encontar la manera de ejecutar cada una de las clases para cada archivo .py teniendo las mismas
        # matrices y vectores bias
        # Revisar la función DataLoader
        outputs = []
        # Habrá que poner el tensor directamente
        targets = []
        for file in folder:
            data = file
            # We need a way to label data
            target = 1
            targets.append(target)

            # Calculate the vector representation for each file .py
            ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code = vector_representation_method(data, feature_size)

            ## forward 
            output = forward(coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code)

            # output append
            # Mirar como concatenar tensores
            if outputs == []:
                outputs = softmax(output)
            else:
                outputs.stack(softmax(output))

        # zero the parameter gradients
        print('outputs: \n', outputs)
        optimizer.zero_grad()

        #Convert lists to tensors
        #outputs = torch.tensor(outputs, dtype = torch.float32)
        targets = torch.tensor(targets, dtype = torch.float32)

        # Loss function 
        loss = criterion(outputs, targets)

        # Calculates the derivative
        loss.backward()

        # Update parameters
        optimizer.step()

        #Time
        end = time()

        print('Epoch ', step, ', Time: ', end-start, ', Loss: ', loss)


def vector_representation_method(data, feature_size):
    # Initializing node list, dict list and dict sibling 
    tree = path_to_module(data)
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    ls_nodes = node_position_assign(ls_nodes)
    ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)

    # Initializing vector embeddings
    embed = Embedding(feature_size, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()

    # Calculate the vector representation for each node
    learning_rate = 0.1
    momentum = 0.01
    vector_representation = Vector_representation_algorithm(ls_nodes, dict_ast_to_Node, feature_size, learning_rate, momentum)
    ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

    return ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code


def forward(coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code):
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code)
    ls_nodes = convolutional_layer.convolutional_layer(ls_nodes, dict_ast_to_Node)
    max_pooling_layer.max_pooling(ls_nodes)
    vector = dynamic_pooling.three_way_pooling(ls_nodes, dict_sibling)
    output = hidden_layer.hidden_layer(ls_nodes, vector)

    return output


def layer_and_SGD_inizialitation(feature_size, alpha):
    # Coding layer initialization
    coding_layer = Coding_layer_algorithm(feature_size)
    w_comb1, w_comb2 = coding_layer.initialize_parameters()
    
    # Convolutional layer initialization
    convolutional_layer = Convolutional_layer_algorithm(feature_size)
    w_t_conv, w_l_conv, w_r_conv, b_conv = convolutional_layer.initialize_parameters()

    # Pooling layer initialization
    max_pooling_layer = Max_pooling_layer()
    dynamic_pooling = Dynamic_pooling_layer()

    # Hidden layer initialization
    hidden_layer = Hidden_layer()
    w_hidden, b_hidden  = hidden_layer.initialize_parameters()

    ### SGD
    # Entire parameter set for TBCNN that we should update
    params = [w_comb1, w_comb2, w_t_conv, w_l_conv, w_r_conv, b_conv, w_hidden, b_hidden]
    # Construct the optimizer
    optimizer = torch.optim.SGD(params, lr = alpha)

    return coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, optimizer

#########################################
# SCRIPT

# Pasar una capeta en lugar de un archivo
filepath = sys.argv[1]

### Inicializar todos los parametros
alpha = 0.1
epoch = 10
feature_size = 20

# Initialization of layers and SGD
coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, optimizer = layer_and_SGD_inizialitation(feature_size, alpha)

### Training set
folder = [filepath]

# Training
train(coding_layer, convolutional_layer, max_pooling_layer, dynamic_pooling, hidden_layer, folder, feature_size, optimizer, epoch)





##########################################################
#pruebas
#eliminar al final

'''
for item in ls_nodes:
    #print(item.__class__.__name__)
    #print(len(item.vector))
    #print(item.type)
    #print(item.vector)
    #print(item.combined_vector)
    print(item.y)
    print(item.pool)
    print('------------')
'''

'''
i = 16
nodo = ls_nodes[i]
print('su padre es')
print(nodo.parent)
print('sus hijos son ')
print(nodo.children)
print(nodo.type)

print('Ahora vamos a trabajar con el padre del nodo')
padre = nodo.parent
print(padre.type)
print(padre.parent)

print('Y ahora veamos sus hijos')
print(nodo.children[0])

print('Finalmente, veamos su vector nuevo (despues de aplicar relu)')
print(nodo.new_vector)
print('Y el vector original')
print(nodo.vector)



i = 16
nodo = ls_nodes[i]
print('Muestra el vector')
print(nodo.vector)
print('Muestra el vector combinado')
print(nodo.combined_vector)
'''


'''
tree = path_to_module(filepath)
ls_nodes, dict_ast_to_Node = node_object_creator(tree)
ls_nodes = node_position_assign(ls_nodes)
ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)

n = 20 #poner el valor que queramos aqui
# TODO recibir n como input a través de la terminal

feature_size = 20
embed = Embedding(10, 5, feature_size, 1, ls_nodes, dict_ast_to_Node)
ls_nodes = embed.node_embedding()[:]
#TODO recibir walkLength = 10, windowSize = 5, vector_size(same as feature_size) = 20 y minCount = 1 a través de la terminal


learning_rate = 0.1
momentum = 0.01
vector_representation = Vector_representation_algorithm(ls_nodes, dict_ast_to_Node, feature_size, learning_rate, momentum)
ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()


coding_layer = Coding_layer_algorithm(ls_nodes, dict_ast_to_Node, feature_size, w_l_code, w_r_code, b_code)
ls_nodes, w_comb1, w_comb2 = coding_layer.coding_layer()
#print(w_comb1, w_comb2)



convolutional_layer = Convolutional_layer_algorithm(ls_nodes, dict_ast_to_Node, feature_size)
ls_nodes, w_t_conv, w_l_conv, w_r_conv, b_conv = convolutional_layer.convolutional_layer()



#matrices = MatrixGenerator(ls_nodes, n)
#w, b = matrices.w, matrices.b

# Max pooling step
max_pooling_layer = Max_pooling_layer(ls_nodes)
max_pooling_layer.max_pooling()

# Dynamic pooling algorithm (Three-way pooling)
dynamic_pooling = Dynamic_pooling_layer(ls_nodes, dict_sibling)
hidden_input = dynamic_pooling.three_way_pooling()
print("The hidden input is: ", hidden_input)


###################
#hidden layer

hidden_layer = Hidden_layer(ls_nodes, hidden_input)

output_hidden, w_hidden, b_hidden  = hidden_layer.hidden_layer()

print('output de hidden layer: ', output_hidden)


##################
#softmax

softmax = nn.Sigmoid()

soft_res = softmax(output_hidden)

print('resultado del softmax: ', soft_res)



###################
# BCELoss for binary prediction (0,1)
# BCEWithLogitsLoss includes the softmax function (Sigmoid)
# CrossEntropyLoss for multi-class prediction (0,1,..,N)

loss = nn.BCEWithLogitsLoss()

# We need to introduce the target

output = loss(soft_res, target)

output.backward()
'''