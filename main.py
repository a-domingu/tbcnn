import sys
import gensim
import random

from node_object_creator import *
from Embeddings import Embedding
from matrix_generator import MatrixGenerator


filepath = sys.argv[1]
tree = path_to_module(filepath)
ls_nodes = node_object_creator(tree)

n = 10 #poner el valor que queramos aqui
# TODO recibir n como input a través de la terminal


embed = Embedding(10, 5, 20, 1, ls_nodes)
ls_nodes = embed.node_embedding()[:]



matrices = MatrixGenerator(ls_nodes, n)
w, b = matrices.w, matrices.b

nodes_vector_update(ls_nodes, w, b)



#########################################
#pruebas
#eliminar al final

'''
for item in ls_nodes:
    print(item)
    print(item.type)
    print(item.vector)
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