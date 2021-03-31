import sys

from node_object_creator import *


filepath = sys.argv[1]
tree = path_to_module(filepath)
ls_nodes = node_object_creator(tree)


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
