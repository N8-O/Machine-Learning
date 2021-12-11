#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:09:29 2021

@author: nathan
"""
import numpy as np

class NN:
    def __init__(self, structure,weights=None):
        # structure is a list of the number of nodes in each layer, 0 is input, -1 is output
        self.layers = len(structure)
        self.structure = structure
        self.w = []
        self.dw = []
        self.nodes = []
        self.nodes.append(np.ones((self.structure[0])))
        
        if weights == None:
        # initialize the weights
            for l in range(1,self.layers-1):
                new_w = np.random.normal(0, 1, (self.structure[l] - 1, self.structure[l-1]))
                new_dw = np.zeros((self.structure[l] - 1, self.structure[l-1]))
                self.w.append(new_w)
                self.dw.append(new_dw)
                self.nodes.append(np.ones((self.structure[l])))
            
            new_w = np.random.normal(0,1,(self.structure[-1],self.structure[-2]))
            new_dw = np.zeros((self.structure[-1], self.structure[-2]))        
            
            self.w.append(new_w)
            self.dw.append(new_dw)
            self.nodes.append(np.ones((self.structure[-1])))
            
        else:
            for l in range(0,self.layers-1):
                new_w = weights[l]
                new_dw = np.zeros(np.shape(weights[l]))
                self.w.append(new_w)
                self.dw.append(new_dw)
                self.nodes.append(np.ones((self.structure[l+1])))            
                    
      
    def activation_function(self,nodes):
        new_nodes = 1/(1+(np.e**-nodes))
        return new_nodes
        
    def forward(self, x):
        self.nodes[0] = x
        for l in range(1,self.layers-1):
            new_nodes = np.array([1])
            new_nodes = np.append(new_nodes,np.matmul(self.w[l-1],self.nodes[l-1]))
            new_nodes = self.activation_function(new_nodes)
            new_nodes[0] = 1 # dont pass the first nodes through the activation function...
            self.nodes[l] = new_nodes
        self.nodes[-1] = np.matmul(self.w[-1],self.nodes[-2])
        
    def backward(self, y):
        # back propagation
        cache = self.nodes[-1]-y
        self.dw[-1][0] = cache*self.nodes[-2]
        
        cache = self.w[-1][0][1:len(self.w[-1][0])]*cache
        
        lay = -2
        
        while lay>(-self.layers):
            for i in range(len(cache)):
                cache[i] = cache[i]*(self.nodes[lay][i+1]*(1-self.nodes[lay][i+1]))
                self.dw[lay][i] = cache[i]*self.nodes[lay-1]            
            new_cache = cache.copy()
            for i in range(len(cache)):
                new_cache[i] = cache[0]*self.w[lay][0][i+1]+cache[1]*self.w[lay][1][i+1]
            
            cache = new_cache
            lay = lay-1
            
            
    # def backward(self, y):
    #     # back propagation
    #     cache = self.nodes[-1]-y
    #     self.dn.append(cache)
    #     self.dw[-1][0] = cache*self.nodes[-2]
        
    #     cache = self.w[-1][0][1:len(self.w[-1][0])]*cache

    #     self.dn.append(cache)
        
    #     lay = -2
        
    #     while lay>(-self.layers):
    #         for i in range(len(cache)):
    #             cache[i] = cache[i]*(self.nodes[lay][i+1]*(1-self.nodes[lay][i+1]))
    #             self.dw[lay][i] = cache[i]*self.nodes[lay-1]   
    #         self.dn.append(cache)                
    #         for i in range(len(cache)):
    #             for l in range(len(self.nodes[lay])-1):
    #                 cache[i] = cache[0]*self.w[lay][0][1]+cache[1]*self.w[lay][1][1]
            
    #         lay = lay-1
            
    def training(self):
        # train network here...
        return
            
            
 

struc = np.array([3,3,3,1])
weights = np.array([[-1 , -2, -3],[1,  2, 3]]),np.array([[ -1, -2 ,  -3],[ 1,  2,  3]]),np.array([[-1,  2,  -1.5]])
a = NN([3,3,3,1],weights)

inp = np.array([1,1,1])
y = 1
a.forward(inp)
a.backward(y)

print("Final HW Practice...")
print("back-propagation algorithm:")
print("Weights: ")
print(a.w)
print("Partial Derivatives: ")
print(a.dw)
        
            
        
        