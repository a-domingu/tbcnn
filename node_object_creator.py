import ast
import sys

from node import Node


#We create the AST
def path_to_module(path):
    return ast.parse(open(path).read())

#We create a list with all AST nodes
def node_object_creator(module):
    dict_ast_to_Node = {} #A dict that relates class ast objects to class Node objects.
    module_node = Node(module)
    if not module in dict_ast_to_Node.keys():
        dict_ast_to_Node[module] = module_node
    ls_nodes = [module_node]
    for child in ast.iter_child_nodes(module):
        ls_nodes = node_object_creator_recursive(module, child, ls_nodes, dict_ast_to_Node)
    return ls_nodes, dict_ast_to_Node

#We instanciate each node as a Node class
def node_object_creator_recursive(parent, node, ls_nodes, dict_ast_to_Node):
    new_node = Node(node, parent)
    if not node in dict_ast_to_Node.keys():
        dict_ast_to_Node[node] = new_node
    #dict[ast] = new_node
    #matriz * dict[Nododuplicado.node]
    #seleccionamos un objeto ast aleatorimente del arbol podriamos hacer dict[random_ast_object]
    #new_node.node gives the AST object
    ls_nodes.append(new_node)
    #new_node.set_index =ls.index(new_node)
    for child in ast.iter_child_nodes(node):
        ls_nodes = node_object_creator_recursive(node, child, ls_nodes, dict_ast_to_Node)
    return ls_nodes

def nodes_vector_update(ls_nodes, w, b):
    for node in ls_nodes:
        node.update_vector(w, b)