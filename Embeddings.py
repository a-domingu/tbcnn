from gensim.models import Word2Vec
import ast
import os
import random

class Embedding:

    def __init__(self, walkLength, windowSize, size, minCount):
        self.walkLength = walkLength
        self.window = windowSize
        self.size = size
        self.minCount = minCount

    #def arr2str(arr):
        #result = ""
        #for i in arr:
            #result += " "+str(i)
        #return result

    def generateWalkFile(self):
        tree = ast.parse(open('observer_2.py').read())
        NodeList = []
        for node in ast.walk(tree):
            walk = self.randomWalk(node)
            NodeList.append(walk)
        return NodeList


    def randomWalk(self, node):
        walkList= []
        curNode = node

        while(len(walkList) < self.walkLength):
            walkList.append(curNode.__class__.__name__)
            curNode = next(ast.iter_child_nodes(curNode))
        return walkList
        
        
    def saveVectors(self, vectors):
        output = open('WalkList.txt', 'w')
        
        output.write(str(len(vectors)) +"\n")
        for i in range(len(vectors)):
            for j in vectors[i]:
                output.write('\t'+ str(j))
            output.write('\n')
        output.close()
        
        
    def node_embedding(self):
        sentenceList = self.generateWalkFile()
        model = Word2Vec(sentenceList, min_count = self.minCount, size = self.size, window = self.window)
        print(model + "hello")
        saveVectors(list(model))


embed = Embedding(10, 5, 20, 1)
embed.node_embedding()