import ast
import sys

from node import Node




def path_to_module(path):
    return ast.parse(open(path).read())

def node_object_creator(module):
    module_node = Node(module)
    ls_nodes = [module_node]
    for child in ast.iter_child_nodes(module):
        ls_nodes = node_object_creator_recursive(module, child, ls_nodes)
    return ls_nodes

def node_object_creator_recursive(parent, node, ls_nodes):
    new_node = Node(node, parent)
    ls_nodes.append(new_node)
    for child in ast.iter_child_nodes(node):
        ls_nodes = node_object_creator_recursive(node, child, ls_nodes)
    return ls_nodes







