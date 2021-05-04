import numpy as np
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

class First_neural_network():
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
        self.vector_matrix = None
        self.vector_p = None
        self.w_l_params = None
        self.w_r_params = None 
        self.node_list = None
        self.l_vector = []


    def vector_representation(self, l2_penalty):
        # Parameters initialization
        self.w_l = torch.randn(self.features_size, self.features_size, requires_grad = True)
        self.w_r = torch.randn(self.features_size, self.features_size, requires_grad = True)
        self.b = torch.randn(self.features_size,  requires_grad = True) 

        ### SGD
        # params is a tensor with vectors (p -> node.vector and node childs c1,..,cN -> node_list), w_r, w_l and b
        params = [node.vector for node in self.ls]
        params.append(self.w_l)
        params.append(self.w_r)
        params.append(self.b)
        # Construct the optimizer
        # Stochastic gradient descent with momentum algorithm
        optimizer = torch.optim.SGD(params, lr = self.alpha, momentum = self.epsilon)

        for step in range(10):
            # Training loop (forward step)
            output_J = self.training_iterations()

            # Computes the cost function (loss)
            loss = self.cost_function_calculation(output_J, l2_penalty)

            # Calculates the derivative
            loss.backward() #self.w_l.grad = dloss/dself.w_l
            
            # Update parameters
            optimizer.step() #self.w_l = self.w_l - lr * self.w_l.grad

            # Zero gradients
            optimizer.zero_grad()

            if (step+1) % 5 == 0:
                print('Epoch: ', step, ' Loss: ', loss)
        
        for node in self.ls:
            node.vector.detach()

        return self.ls, self.w_l.detach(), self.w_r.detach(), self.b.detach()


    # We applied the coding criterion for each non-leaf node p in AST
    def training_iterations(self):
        sum_error_function = torch.tensor([0])
        for node in self.ls:
            if len(node.children) > 0:
                # Generates training sample and computes d 
                self.training_sample_d(node)
                d = self.coding_criterion_d(node)
                # Generates a negative sample and computes d_c
                self.negative_sample_d_c(node)
                d_c = self.coding_criterion_d(node)
                # Computes the error function J(d,d_c) for each node and computes the sum
                sum_error_function = sum_error_function + self.error_function_J(d_c, d)       
        return sum_error_function

    
    def training_sample_d(self, node):
        # We save the vector p
        self.vector_p = node.vector
        # child is an AST object and we convert the AST object to a Node object
        # child list is a matrix with all the child's vectors
        vectors = []
        vector_l = []
        for child in node.children:
            vectors.append(self.dict_ast_to_Node[child].vector)
            vector_l.append((self.dict_ast_to_Node[child].leaves_nodes/node.leaves_nodes))
        # We create a matrix with all the vectors
        self.vector_matrix = torch.stack(tuple(vectors), 0)
        # We create a 3D tensor for the vector matrix: shape(nb_nodes, 30, 1)
        #print('vector matrix shape: ', self.vector_matrix.shape)
        self.vector_matrix = self.vector_matrix.unsqueeze(2)
        #print('vector matrix shape: ', self.vector_matrix.shape)
        #print('vector matrix: ', self.vector_matrix)
        
        # We create a vector with all the l_i values
        self.l_vector = torch.tensor(vector_l)
        # We create a 3D tensor for the vector with all l_i values: shape(nb_nodes, 1, 1)
        self.l_vector = self.l_vector.unsqueeze(1)
        self.l_vector = self.l_vector.unsqueeze(1)
        #print('l_vector shape: ', self.l_vector.shape)

    
    def negative_sample_d_c(self, node):
        # We choose a Node class that cames from a ls_nodes    
        symbol = random.choice(self.ls)
        # We substitutes randomly a vector with a different vector
        index = random.randint(0,len(self.l_vector))
        if index == 0:
            self.vector_p = symbol.vector
        else:
            self.vector_matrix[index-1, :, :] = symbol.vector.unsqueeze(1)
            self.l_vector[index-1, :, :] = symbol.leaves_nodes


    # Compute the cost function (function objective)
    def cost_function_calculation(self, sum_J, l2_penalty):
        first_term = (1/len(self.ls)*sum_J)
        # Norms calculations
        norm_w_l = torch.norm(self.w_l, p='fro')
        squared_norm_w_l = norm_w_l * norm_w_l
        norm_w_r = torch.norm(self.w_r, p='fro')
        squared_norm_w_r = norm_w_r * norm_w_r
        # Second term calculation(Revisar el parametro lambda del paper!!!)
        second_term = (l2_penalty/(2*2*self.features_size*self.features_size))*(squared_norm_w_l + squared_norm_w_r)
        return first_term + second_term


    # Calculate the error function J(d,d_c)
    def error_function_J(self, d_c, d):
        margin = torch.tensor([1])
        error_function = margin + d - d_c
        return max(torch.tensor([0]), error_function)


    # Calculate the square of the Euclidean distance between the real vector and the target value.
    def coding_criterion_d(self, node):
        # Calculate the target value
        if self.vector_matrix.shape[0] > 1:
            calculated_vector = self.calculate_vector(node)
        elif self.vector_matrix.shape[0] == 1:
            calculated_vector = self.calculate_vector_special_case()

        # Calculate the square of the Euclidean distance, d
        diff_vector = self.vector_p - calculated_vector
        euclidean_distance = torch.norm(diff_vector, p=2)
        d = euclidean_distance * euclidean_distance
        return d


    # Calculate the target value
    def calculate_vector(self, node):
        # Parameters used to calculate the weight matrix for each node: shape(nb_nodes, 1, 1)
        n = self.vector_matrix.shape[0]
        w_l_list = []
        w_r_list = []
        for i in range(1,n+1):
            w_l_list.append((n-i)/(n-1))
            w_r_list.append((i-1)/(n-1))
        self.w_l_params = torch.tensor(w_l_list)
        self.w_r_params = torch.tensor(w_r_list)
        #print('w_l params: ', self.w_l_params)
        #print('Number of rows: ', self.w_l_params.shape)
        self.w_l_params = self.w_l_params.unsqueeze(1)
        self.w_l_params = self.w_l_params.unsqueeze(1)
        #print('Number of rows: ', self.w_l_params.shape)
        #print('w_r params: ', self.w_r_params)
        #print('Number of rows: ', self.w_r_params.shape)
        self.w_r_params = self.w_r_params.unsqueeze(1)
        self.w_r_params = self.w_r_params.unsqueeze(1)
        #print('Number of rows: ', self.w_r_params.shape)

        # We create a 3D tensor for left matrix and right matrix: shape(nb_nodes, 30, 30)
        #print('w_l shape: ', self.w_l.shape)
        #print('w_r shape: ', self.w_r.shape)
        '''
        print('w_l shape: ', self.w_l.shape)
        self.w_l = torch.stack(tuple(self.w_l for i in range(0,n)), 0)
        print('w_l shape: ', self.w_l.shape)
        print('w_r shape: ', self.w_r.shape)
        self.w_r = torch.stack(tuple(self.w_r for i in range(0,n)), 0)
        print('w_r shape: ', self.w_r.shape)
        left_matrix = self.w_l_params*self.w_r
        #print('left matrix: ', left_matrix)
        print('left matrix shape: ', left_matrix.shape)
        right_matrix = self.w_r_params*self.w_r
        #print('right matrix: ', right_matrix)
        print('right matrix shape: ', right_matrix.shape)
        print('Shape of the matrix of vectors: ', self.vector_matrix.shape)
        '''
        # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
        # We compute the weighted matrix: shape(nb_nodes, 30, 30)
        weighted_matrix = self.l_vector*((self.w_l_params*self.w_l)+(self.w_r_params*self.w_r))
        #print('Weighted matrix: ', weighted_matrix)
        #print('Weighted matrix shape: ', weighted_matrix.shape)

        # Sum the weighted values over vec(·)
        final_matrix = torch.matmul(weighted_matrix, self.vector_matrix)
        final_vector = torch.sum(final_matrix, 0)
        final_vector = final_vector.squeeze(1)
        #print('Vector: ', final_vector)
        #print('Vector shape: ', final_vector.shape)
        #print('Vector b shape: ', self.b.shape)
        return F.relu(final_vector + self.b, inplace=False)


    # Calculate the weighted matrix for a node with only one child
    def calculate_vector_special_case(self):
        matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
        print('Vector shape: ', self.vector_matrix.shape)
        vector = self.vector_matrix
        vector = vector.squeeze(2)
        vector = vector.squeeze(0)
        print('New vector shape: ', vector.shape)
        final_vector = torch.matmul(matrix, vector) + self.b
        return F.relu(final_vector, inplace=False)
