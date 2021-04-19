import sys
import gensim
import random
from torch import nn

from node_object_creator import *
from embeddings import Embedding
from node import Node
from matrix_generator import MatrixGenerator
from vector_representation import Vector_representation_algorithm
from coding_layer import Coding_layer_algorithm
from convolutional_layer import Convolutional_layer_algorithm
from pooling_layer import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer


filepath = sys.argv[1]
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


'''
###################
# BCELoss for binary prediction (0,1)
# CrossEntropyLoss for multi-class prediction (0,1,..,N)

loss = nn.BCELoss()

# We need to introduce the target

output = loss(soft_res, target)

output.backward()
'''

#####################################
'''
softmax = nn.Sigmoid()
### SGD
# Entire parameter set for TBCNN that we should update
params = [w_comb1, w_comb2, w_t_conv, w_l_conv, w_r_conv, b_conv, w_hidden, b_hidden]
# Construct the optimizer
optimizer = torch.optim.SGD(params, lr = alpha)
def train(softmax, data, optimizer, epoch):
    """Create the training loop"""
    # We need to create a way to run all layers each time (coding_layer, convolutional_layer, hidden_layer).
    features = data['features']
    # We need a way to label data
    target = data['target']
    criterion = nn.BCELoss()

    for step in range(epoch):
        # zero the parameter gradients
        optimizer.zero_grad()
        ## forward + backward + optimize
        # forward step
        # We can change the softmax by a function that calculates all forward steps until softmax
        outputs = softmax(output_hidden)
        # Loss function 
        loss = criterion(outputs, target)
        # Calculates the derivative
        loss.backward()
        # Update parameters
        optimizer.step()
'''




#########################################
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