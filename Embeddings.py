import ast
import os
import random

from gensim.models import Word2Vec
from node_object_creator import *
class Embedding():

    def __init__(self, walkLength, windowSize, size, minCount, ls_nodes):
        self.walkLength = walkLength
        self.window = windowSize
        self.size = size
        self.minCount = minCount
        self.ls = ls_nodes
        self.embedding = self.node_embedding()


    def generateWalkFile(self):
        walkMatrix = []
        for node in self.ls:
            walk = self.randomWalk(node)
            walkMatrix.append(walk)
        return walkMatrix


    def randomWalk(self, node):
        walkList= []
        
        while(len(walkList) < self.walkLength):
            walkList.append(str(node.type))
            if node.children: 
                node = random.choice(node.children)
            else:
                break
        return walkList
        
        
    def saveVectors(self, model):
        for node in self.ls:
            vector = model.wv[node.type]
            node.set_vector(vector)
            #print(model.wv[node.type])


    def node_embedding(self):
        matrix = self.generateWalkFile()
        model = Word2Vec(matrix, min_count = self.minCount, vector_size = self.size, window = self.window)
        #print(model)
        #words = list(model.wv.index_to_key)
        #print(words)
        #vec_load = model.wv['Load']
        #print(vec_load)
        self.saveVectors(model)
        return self.ls
