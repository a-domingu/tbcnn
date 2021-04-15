import ast
import sys

from node import Node


#We create the AST
def path_to_module(path):
    return ast.parse(open(path).read())

#We create a list with all AST nodes
def node_object_creator(module):
    dict_ast_to_Node = {} #A dict that relates class ast objects to class Node objects.
    depth = 1
    module_node = Node(module, depth)
    if not module in dict_ast_to_Node.keys():
        dict_ast_to_Node[module] = module_node
    ls_nodes = [module_node]
    for child in ast.iter_child_nodes(module):
        ls_nodes = node_object_creator_recursive(module, child, ls_nodes, dict_ast_to_Node, depth)
    return ls_nodes, dict_ast_to_Node

#We instanciate each node as a Node class
def node_object_creator_recursive(parent, node, ls_nodes, dict_ast_to_Node, depth):
    depth += 1
    new_node = Node(node, depth, parent)
    if not node in dict_ast_to_Node.keys():
        dict_ast_to_Node[node] = new_node

    ls_nodes.append(new_node)
    for child in ast.iter_child_nodes(node):
        ls_nodes = node_object_creator_recursive(new_node, child, ls_nodes, dict_ast_to_Node, depth)
    return ls_nodes

def nodes_vector_update(ls_nodes, w, b):
    for node in ls_nodes:
        node.update_vector(w, b)

def node_position_assign(ls_nodes):
    for index in range(len(ls_nodes)):
        count  = 1
        good_depth = ls_nodes[index].depth
        for node in ls_nodes[0:index]:
            if node.depth == good_depth:
                count+=1
        ls_nodes[index].set_position(count)
        #print("Para el nodo", index, "tenemos depth", ls_nodes[index].depth, "y position", ls_nodes[index].position)
    
    return ls_nodes
