import pytest
import numpy as np
import ast
import torch
from gensim.models import Word2Vec

from embeddings import Embedding
from node import Node
from matrix_generator import MatrixGenerator
from node_object_creator import *
from vector_representation import vector_representation_algorithm
from coding_layer import coding_layer_algorithm


@pytest.fixture
def set_up_dictionary():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    return tree, dict_ast_to_Node

@pytest.fixture
def set_up_embeddings():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    return embed

@pytest.fixture
def set_up_matrix():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    matrices = MatrixGenerator(ls_nodes, 10)
    return matrices

@pytest.fixture
def set_up_update_vector():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    matrices = MatrixGenerator(ls_nodes, 20)
    w, b = matrices.w, matrices.b
    nodes_vector_update(ls_nodes, w, b)
    w, b = matrices.w, matrices.b
    nodes_vector_update(ls_nodes, w, b)
    return ls_nodes

@pytest.fixture
def set_up_vector_representation():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = vector_representation_algorithm(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    return ls_nodes, w_l, w_r, b_code

@pytest.fixture
def set_up_coding_layer():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = vector_representation_algorithm(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    coding_layer = coding_layer_algorithm(ls_nodes, dict_ast_to_Node, 20, w_l, w_r, b_code)
    ls_nodes, w_comb1, w_comb2 = coding_layer.coding_layer()
    return ls_nodes, w_comb1, w_comb2



def test_dictionary_Node(set_up_dictionary):

    tree, dict_ast_to_Node = set_up_dictionary

    for node in ast.iter_child_nodes(tree):
        assert node in dict_ast_to_Node
        assert dict_ast_to_Node[node].__class__.__name__ == "Node"

# Error a solucionar (file Embeddings.py line 37)
# Hay un error porque no reconoce como argumento "vector_size" en el comando word2vec. 
# Al escribir "size" como argumento funcionan los test pero da error al ejecutar el código. 
# Viceversa cuando escribimos "vector_size" funciona el código pero da error en los tests.
def test_node_embedding(set_up_embeddings):

    result = set_up_embeddings.node_embedding()[:]
    length_expected = 20

    for el in result:
        assert len(el.vector) == length_expected
                        
def test_matrix_length(set_up_matrix):
    
    w, b = set_up_matrix.w, set_up_matrix.b

    assert w.shape == (10, 20)
    assert len(b) == 10

def test_update_vector(set_up_update_vector):

    for node in set_up_update_vector:
        assert len(node.new_vector) > 0

def test_vector_representation(set_up_vector_representation):
    
    ls_nodes, w_l, w_r, b_code = set_up_vector_representation
    feature_size_expected = 20

    for node in ls_nodes:
        assert len(node.vector) == feature_size_expected
        vector = node.vector.detach().numpy()
        assert np.count_nonzero(vector) != 0
    
    assert w_l.shape == (feature_size_expected, feature_size_expected)
    w_l = w_l.numpy()
    assert np.count_nonzero(w_l) != 0
    assert w_r.shape == (feature_size_expected, feature_size_expected)
    w_r = w_r.numpy()
    assert np.count_nonzero(w_r) != 0
    assert len(b_code) == feature_size_expected

def test_coding_layer(set_up_coding_layer):
    
    ls_nodes, w_comb1, w_comb2 = set_up_coding_layer
    feature_size_expected = 20

    for node in ls_nodes:
        assert len(node.combined_vector) == feature_size_expected
        vector = node.combined_vector.detach().numpy()
        assert np.count_nonzero(vector) != 0
    
    assert w_comb1.shape == (feature_size_expected, feature_size_expected)
    w_comb1 = w_comb1.detach().numpy()
    assert np.count_nonzero(w_comb1) != 0
    assert w_comb2.shape == (feature_size_expected, feature_size_expected)
    w_comb2 = w_comb2.detach().numpy()
    assert np.count_nonzero(w_comb2) != 0
