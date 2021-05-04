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
        vectors = tuple(node.vector for node in self.ls)
        self.vector_matrix = torch.stack(vectors, 0)
        print('Shape of the matrix of vectors: ', self.vector_matrix.shape)
            

        ### SGD
        # params is a tensor with vectors (p -> node.vector and node childs c1,..,cN -> node_list), w_r, w_l and b
        params = [node.vector for node in self.ls]
        params.append(self.w_l)
        params.append(self.w_r)
        params.append(self.b)
        # Construct the optimizer
        # Stochastic gradient descent with momentum algorithm
        optimizer = torch.optim.SGD(params, lr = self.alpha, momentum = self.epsilon)

        for step in range(100):
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


    def negative_sample_d_c(self, node):
        # We save the vector p
        self.vector_p = node.vector
        # child is an AST object and we convert the AST object to a Node object
        # child list is a matrix with all the child's vectors
        # We create a matrix with all the vectors
        self.vector_matrix = torch.stack(tuple(self.dict_ast_to_Node[child].vector for child in node.children), 0)
        # We create a vector with all the l_i values
        self.l_vector = torch.tensor(tuple((self.dict_ast_to_Node[child].leaves_nodes/node.leaves_nodes) for child in node.children))
        print('Vector matrix: ', self.vector_matrix)
        print('l vector: ', self.l_vector)

        # We choose a Node class that cames from a ls_nodes    
        symbol = random.choice(self.ls)
        # We substitutes randomly a vector with a different vector
        index = random.randint(0,len(self.l_vector))
        if index == 0:
            self.vector_p = symbol.vector
        else:
            self.vector_matrix[index-1] = symbol.vector
            self.l_vector[index-1] = symbol.leaves_nodes
        print('index: ', index)
        print('Vector matrix: ', self.vector_matrix)
        print('l vector: ', self.l_vector)


    def training_sample_d(self, node):
        # We create a matrix with all the vectors
        self.vector_p = node.vector
        # We create a vector with all the l_i values
        #self.l_vector = [(node.leaves_nodes/node.leaves_nodes)]
        # child is an AST object and we convert the AST object to a Node object
        # child list is a matrix with all the child's vectors
        self.vector_matrix = torch.stack(tuple(self.dict_ast_to_Node[child].vector for child in node.children), 0)
        # we convert a vector into a matrix 
        #self.vector_matrix = torch.unsqueeze(self.vector_matrix, 0)
        # We concat all the vectors 
        #self.vector_matrix = torch.cat((self.vector_matrix, child_list), 0)
        # We save all the l_i values in a vector
        self.l_vector = torch.tensor(tuple((self.dict_ast_to_Node[child].leaves_nodes/node.leaves_nodes) for child in node.children))
        print('Vector matrix: ', self.vector_matrix)
        print('l vector: ', self.l_vector)


    # Calculate the square of the Euclidean distance between the real vector and the target value.
    def coding_criterion_d(self, node):
        print('Vector matrix: ', self.vector_matrix)
        print('Number of rows: ', self.vector_matrix.shape[0])
        print('l vector: ', self.l_vector)

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
        # initalize the target value array
        sum = torch.zeros(self.features_size)
        # Parameters used to calculate the weight matrix for each node
        n = self.vector_matrix.shape[0]
        self.w_l_params = torch.tensor(tuple((n-i)/(n-1) for i in range(1,n)))
        print('w_l params: ', self.w_l_params)
        print('Number of rows: ', self.w_l_params.shape[0])
        self.w_r_params = torch.tensor(tuple((i-1)/(n-1) for i in range(1,n)))


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
            sum = sum + torch.matmul(matrix, child.vector)
            i += 1
        return F.relu(sum + self.b)


    # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
    def weight_matrix_update(self, n, i):
        # Hay que crear un tensor con tamaño (nb_nodes, 1, 1) y guardar (n-i)/(n-1) y luego otro tensor con todo w_l (nb_nodes, 30, 30). Multiplicar ambos tensores con z_1 = y_2*w_l
        # Lo mismo para la matriz derecha 
        left_matrix = ((n-i)/(n-1))* self.w_l
        right_matrix = ((i-1)/(n-1))*self.w_r
        return (left_matrix + right_matrix) 


    # Calculate the weighted matrix for a node with only one child
    def calculate_vector_special_case(self):
        matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
        vector = torch.matmul(matrix, self.vector_matrix) + self.b
        return F.relu(vector)
