import ast
import sys

from node import Node


def path_to_module(path):
    return ast.parse(open(path).read())

def node_object_creator(module):
    module_node = Node(module)
    ls_nodes = [module_node]
    for child in ast.iter_child_nodes(module):
        ls_nodes = node_object_creator_recursive(module_node, child, ls_nodes)
    return ls_nodes

def node_object_creator_recursive(parent, node, ls_nodes):
    new_node = Node(node, parent)
    ls_nodes.append(new_node)
    for child in ast.iter_child_nodes(node):
        ls_nodes = node_object_creator_recursive(new_node, child, ls_nodes)
    return ls_nodes

def nodes_vector_update(ls_nodes, w, b):
    for node in ls_nodes:
        node.update_vector(w, b)







