import sys
import gensim
import random

from node_object_creator import *
from Embeddings import Embedding
from matrix_generator import MatrixGenerator


filepath = sys.argv[1]
tree = path_to_module(filepath)
ls_nodes = node_object_creator(tree)
embed = Embedding(10, 5, 20, 1, ls_nodes)
embed.node_embedding()

#########################################
#pruebas
#eliminar al final


#for item in ls_nodes:
#    print(item)
#    print(item.type)

#i = 16
#nodo = ls_nodes[i]
#print('su padre es')
#print(nodo.parent)
#print('sus hijos son ')
#print(nodo.children)
#print(nodo.type)

n = 10 #poner el valor que queramos aqui
# TODO recibir n como input a través de la terminal


matrices = MatrixGenerator(ls_nodes, n)
w, b = matrices.w, matrices.b

nodes_vector_update(ls_nodes, w, b)