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
        self.w_l = None
        self.w_r = None
        self.b = None
        self.stop_criteria = 0.01
        self.node_list = []


    def gradient_descent(self):
        matrices = MatrixGenerator(self.ls, self.features_size)
        self.w_l = matrices.w
        self.w_r = matrices.w
        self.b = matrices.b
        loss = 1

        while loss > self.stop_criteria:
            loss = self.training_iterations()

        return self.ls, self.w_l, self.w_r, self.b

    def training_iterations(self):
        loss = 0
        for node in self.ls:
            if len(node.children) > 0:
                self.negative_sample_d_c(node)
                d_c = self.coding_criterion_d()
                self.training_sample_d(node)
                d = self.coding_criterion_d()
                loss = loss + self.error_function_J(d_c, d)
        return loss


    def error_function_J(self, d_c, d):
        margin = 1
        error_function = margin + d - d_c
        return max(0, error_function)


    def negative_sample_d_c(self, node):
        self.node_list = []
        self.node_list.append(node)

        for child in node.children:
            self.node_list.append(child)
        
        symbol = random.choice(self.ls)
        index = random.randint(0,len(self.node_list))
        self.node_list[index] = symbol 

    def training_sample_d(self, node):
        self.node_list = []
        self.node_list.append(node)

        for child in node.children:
            self.node_list.append(child)   


    def coding_criterion_d(self):
        node = self.node_list.pop(0)
        if len(self.node_list) > 1:
            calculated_vector = self.calculate_vector(node)
        elif len(node.children) == 1:
            calculated_vector = self.calculate_vector_special_case()

        diff_vector = node.vector - calculated_vector
        euclidean_distance = np.linalg.norm(diff_vector)
        d = euclidean_distance * euclidean_distance
        return d


    def calculate_vector(self, node):
        '''
        This function's output is the vector vec(·) of each node,
        following the stochastic Gradient Descent with Momentum algorithm of the
        "Building Program Vector Representations for Deep Learning" report
        '''
        sum = np.zeros(self.features_size)
        n = len(node.children)
        if n==0:
            n = len(self.node_list)
        i=1
        for child in self.node_list:
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


    def calculate_vector_special_case(self):
        for child in self.node_list:
            matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
            vector = np.dot(matrix, child.vector) + self.b
        return relu(vector)
