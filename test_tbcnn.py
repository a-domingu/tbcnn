import pytest
import numpy as np
import ast
import torch
from gensim.models import Word2Vec

from Embeddings import Embedding
from node import Node
from matrix_generator import MatrixGenerator
from node_object_creator import *

@pytest.fixture
def set_up_embeddings():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embedding = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    return embedding

@pytest.fixture
def set_up_nodes():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    return ls_nodes

@pytest.fixture
def set_up_matrix():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embedding = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embedding.node_embedding()[:]
    matrices = MatrixGenerator(ls_nodes, 10)
    return matrices

@pytest.fixture
def set_up_update_vector():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embedding = Embedding(10, 5, 20, 1, ls_nodes, dict_ast_to_Node)
    ls_nodes = embedding.node_embedding()[:]
    matrices = MatrixGenerator(ls_nodes, 10)
    w, b = matrices.w, matrices.b
    w = torch.tensor(w)
    b = torch.tensor(b)
    nodes_vector_update(ls_nodes, w, b)
    w, b = matrices.w, matrices.b
    w = torch.tensor(w)
    b = torch.tensor(b)
    nodes_vector_update(ls_nodes, w, b)
    return ls_nodes

#Error a solucionar
#Hay un error porque no reconoce como argumento "vector_size" en el comando word2vec. 
#Al escribir "size" como argumento funcionan los test pero da error al ejecutar el código
def test_node_embedding(set_up_embeddings):

    result = set_up_embeddings.node_embedding()[:]
    length_expected = 20

    for el in result:
        assert len(el.vector) == length_expected

def test_node_context(set_up_nodes):

    for node in set_up_nodes:
        children_result = node.children
        if children_result:
            for child in children_result:
                assert child.__class__.__name__ == "Node"
                        
def test_matrix_length(set_up_matrix):
    
    w, b = set_up_matrix.w, set_up_matrix.b

    assert w.shape == (10, 20)
    assert len(b) == 10

def test_update_vector(set_up_update_vector):

    for node in set_up_update_vector:
        assert len(node.new_vector) > 0