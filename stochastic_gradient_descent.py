import numpy as np
import random

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

class stochastic_gradient_descent_momentum():

    def __init__(self, ls_nodes, features_size, learning_rate, momentum):
        self.ls = ls_nodes
        self.features_size = features_size
        self.alpha = learning_rate
        self.epsilon = momentum
        self.w_l = self.initalize_weight_matrix()
        self.w_r = self.initalize_weight_matrix()
        self.b = self.initalize_bias_vector()


    def initalize_weight_matrix(self):
        matrices = MatrixGenerator(self.ls, self.features_size)
        weight_matrix = matrices.w
        return weight_matrix

    def initalize_bias_vector(self):
        matrices = MatrixGenerator(self.ls, self.features_size)
        bias_vector = matrices.b
        return bias_vector

    def calculate_vector(self, node):
        '''
        This function's output is the vector vec(·) of each node,
        following the stochastic Gradient Descent with Momentum algorithm of the
        "Building Program Vector Representations for Deep Learning" report
        '''
        sum = np.zeros(self.features_size)
        n = len(node.children)
        i=1
        for child in node.children:
            weighted_matrix = self.weight_matrix_update(n, i)
            l = (child.get_l/node.get_l)
            matrix = np.dot(l, weighted_matrix)
            sum = sum + np.dot(matrix, child.vector) 
            i += 1
        return relu(sum + self.b)

    def weight_matrix_update(self, n, i):
        left_matrix = ((n-i)/(n-1))*self.w_l 
        right_matrix = ((i-1)/(n-1))*self.w_r
        return (left_matrix + right_matrix) 

    def calculate_vector_special_case(self, node):
        for child in node.children:
            matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
            vector = np.dot(matrix, child.vector) + self.b
        return relu(vector)

    def coding_criterion_d(self, node):
        if len(node.children) > 1:
            calculated_vector = self.calculate_vector(node)
        elif len(node.children) == 1:
            calculated_vector = self.calculate_vector_special_case(node)

        diff_vector = node.vector - calculated_vector
        euclidean_distance = np.linalg.norm(diff_vector)
        d = euclidean_distance * euclidean_distance
        return d
    
    def compute_J(self, node):
        margin = 1
        error_function = margin + self.coding_criterion_d(node)
        return max(0, error_function)

    def gradient_descent(self):
        for node in self.ls:
            max = self.compute_J(node)