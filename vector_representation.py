import numpy as np
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

class Vector_representation_algorithm():
    '''
    In this class we update vec(·), where vec(·) is the feature representation of a node in the AST.
    We use the stochastic gradient descent with momentum algorithm of the 
    "Building Program Vector Representations for Deep Learning" report.
    First we compute the cost function J by using the coding criterion d and then we applied the back
    propagation algorithm

    Inputs:
    ls_nodes [list <class Node>]: list with all nodes in the AST
    dict_ast_to_Node[dict[ast_object] = <class Node>]: dictionary that relates class ast objects to class Node objects
    features_size [int]: Vector embedding size
    learning_rate [int]: learning rate parameter 'alpha' used in the SGD algorithm
    momentum [int]: momentum parameter 'epsilon' used in the SGD with momentum algorithm

    Output:
    ls_nodes [list <class Node>]: We update vector embedding of all nodes
    w_l [matrix[features_size x features_size]]: left weight matrix used as parameter
    w_r [matrix[features_size x features_size]]: right weight matrix used as parameter
    b [array[features_size]]: bias term
    '''

    def __init__(self, ls_nodes, dict_ast_to_Node, features_size, learning_rate, momentum):
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node
        self.features_size = features_size
        self.alpha = learning_rate
        self.epsilon = momentum
        self.w_l = None
        self.w_r = None
        self.b = None
        self.stop_criteria = 1
        self.node_list = []


    def vector_representation(self):
        # Parameters initialization
        matrices = MatrixGenerator(self.features_size, self.features_size)
        self.w_l = matrices.w
        self.w_r = matrices.w
        self.b = matrices.b

        ### SGD
        # params is a tensor with vectors (p -> node.vector and node childs c1,..,cN -> node_list), w_r, w_l and b
        params = [node.vector for node in self.ls]
        params.append(self.w_l)
        params.append(self.w_r)
        params.append(self.b)
        # Construct the optimizer
        # Stochastic gradient descent with momentum algorithm
        optimizer = torch.optim.SGD(params, lr = self.alpha, momentum = self.epsilon)

        loss = 1000
        while loss > self.stop_criteria:
            # Training loop (forward step)
            output_J = self.training_iterations()

            # Computes the cost function (loss)
            loss = self.cost_function_calculation(output_J)

            # Calculates the derivative
            loss.backward()

            # Update parameters
            optimizer.step()
            # Set the updates vectors
            for node in self.ls:
                node.set_vector(node.vector)

            # Zero gradients
            optimizer.zero_grad()

        return self.ls, self.w_l.detach(), self.w_r.detach(), self.b.detach()


    # We applied the coding criterion for each non-leaf node p in AST
    def training_iterations(self):
        sum_error_function = torch.tensor([0])
        for node in self.ls:
            if len(node.children) > 0:
                # Generates a negative sample and computes d_c
                self.negative_sample_d_c(node)
                d_c = self.coding_criterion_d(node)
                # Generates training sample and computes d
                self.training_sample_d(node)
                d = self.coding_criterion_d(node)
                # Computes the error function J(d,d_c) for each node and computes the sum
                sum_error_function = sum_error_function + self.error_function_J(d_c, d)       
        return sum_error_function


    # Compute the cost function (function objective)
    def cost_function_calculation(self, sum_J):
        first_term = (1/len(self.ls)*sum_J)
        # Norms calculations
        norm_w_l = torch.norm(self.w_l, p='fro')
        squared_norm_w_l = norm_w_l * norm_w_l
        norm_w_r = torch.norm(self.w_r, p='fro')
        squared_norm_w_r = norm_w_r * norm_w_r
        # Second term calculation(Revisar el parametro lambda del paper!!!)
        second_term = (1/(2*2*self.features_size*self.features_size))*(squared_norm_w_l + squared_norm_w_r)
        return first_term + second_term


    # Calculate the error function J(d,d_c)
    def error_function_J(self, d_c, d):
        margin = torch.tensor([1])
        error_function = margin + d - d_c
        return max(torch.tensor([0]), error_function)


    def negative_sample_d_c(self, node):
        self.node_list = []
        # node is a Node class that cames from a ls_nodes
        self.node_list.append(node)
        # child is an AST object
        for child in node.children:
            # We convert the AST object to a Node object
            child_node = self.dict_ast_to_Node[child]
            self.node_list.append(child_node)
        # We choose a Node class that cames from a ls_nodes    
        symbol = random.choice(self.ls)
        # We substitutes randomly a node with a different random node
        index = random.randint(0,len(self.node_list)-1)
        self.node_list[index] = symbol 


    def training_sample_d(self, node):
        self.node_list = []
        # node is a Node class that cames from a ls_nodes
        self.node_list.append(node)
        # child is an AST object
        for child in node.children:
            # We convert the AST object to a Node object
            child_node = self.dict_ast_to_Node[child]
            self.node_list.append(child_node)   


    # Calculate the square of the Euclidean distance between the real vector and the target value.
    def coding_criterion_d(self, node):
        node_vector = self.node_list.pop(0)

        # Calculate the target value
        if len(self.node_list) > 1:
            calculated_vector = self.calculate_vector(node)
        elif len(node.children) == 1:
            calculated_vector = self.calculate_vector_special_case()

        # Calculate the square of the Euclidean distance, d
        diff_vector = node_vector.vector - calculated_vector
        euclidean_distance = torch.norm(diff_vector, p=2)
        d = euclidean_distance * euclidean_distance
        return d

    # Calculate the target value
    def calculate_vector(self, node):
        # initalize the target value array
        sum = torch.zeros(self.features_size)
        # Parameters used to calculate the weight matrix for each node
        n = len(node.children)
        i=1
        # number of leaves nodes under node p
        l_p = self.get_l(node)
        # Sum the weighted values over vec(·)
        for child in self.node_list:
            # The weighted matrix for each node is a linear combination of matrices w_l and w_r
            weighted_matrix = self.weight_matrix_update(n, i)
            # number of leaves nodes under child node
            l_c = self.get_l(child)
            l = (l_c/l_p)
            # The weighted matrix is weighted by the number of leaves nodes under child node
            matrix = l*weighted_matrix
            # Sum the weighted values over vec(child)
            sum = sum + matrix*child.vector
            i += 1
        return F.relu(sum + self.b)


    # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
    def weight_matrix_update(self, n, i):
        left_matrix = ((n-i)/(n-1))* self.w_l
        right_matrix = ((i-1)/(n-1))*self.w_r
        return (left_matrix + right_matrix) 


    # Calculate the weighted matrix for a node with only one child
    def calculate_vector_special_case(self):
        for child in self.node_list:
            matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
            vector = matrix*child.vector + self.b
        return F.relu(vector)


    # Calculate the number of leaves nodes under each node
    def get_l(self, node):
        '''
        This function's output is the number of leaf nodes under each node
        '''
        leaves_under_node = 0
        if len(node.children) == 0:
            return leaves_under_node
        else:
            leaves_nodes = self.calculate_l(node, leaves_under_node)
        return leaves_nodes


    def calculate_l(self, node, leaves_under_node):
        #node is a Node object
        #child is an AST object
        for child in node.children:
            child_node = self.dict_ast_to_Node[child]
            if len(child_node.children) == 0:
                leaves_under_node += 1
            else:
                leaves_under_node = self.calculate_l(child_node, leaves_under_node)
        return leaves_under_node
