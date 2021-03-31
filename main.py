import sys

from node_object_creator import *
from matrix_generator import MatrixGenerator


filepath = sys.argv[1]
tree = path_to_module(filepath)
ls_nodes = node_object_creator(tree)

n = 10 #poner el valor que queramos aqui
# TODO recibir n como input a través de la terminal


matrices = MatrixGenerator(ls_nodes, n)
w, b = matrices.w, matrices.b

nodes_vector_update(ls_nodes, w, b)


#########################################
#pruebas
#eliminar al final


for item in ls_nodes:
    print(item)

i = 10
nodo = ls_nodes[i]
print('su padre es')
print(nodo.parent)
print('sus hijos son ')
print(nodo.children)
