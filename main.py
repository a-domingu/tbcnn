import sys
import gensim
import random

from node_object_creator import *
from Embeddings import Embedding


filepath = sys.argv[1]
tree = path_to_module(filepath)
#ls_nodes = node_object_creator(tree)


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

embed = Embedding(10, 5, 20, 1, tree)
embed.node_embedding()