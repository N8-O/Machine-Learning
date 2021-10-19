# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:07:51 2021

@author: Nathan
"""
import time
from BankDataRead import ReadandOrganizeBankData
from ID3_algorithm import *
import numpy as np

# Bagged Decision Trees


def BagTrees(data,labels,test_data,test_labels,attribute_classifiers,label_classifiers,inf_gain,tree_depth,iterations,weights,m):
    Trees = []
    
    for i in range(iterations):
        # generate random selection of data
        if isinstance(m, str):
            m = np.random.choice(range(1,len(labels)))
        
        passed_data = {}
        passed_labels = []
        resampled_index = np.random.choice(range(len(labels)), m)
        for key in data:
            passed_data[key] = [data[key][index] for index in resampled_index]
        passed_labels = [labels[index] for index in resampled_index]
        
        # learn full decision tree on randomly selected data
        Trees.append(ID3(passed_data, attribute_classifiers, passed_labels, label_classifiers, tree_depth, weights, inf_gain))
        
        dict_output = ReturnDictionaryOutput(Trees[i],data,labels)
        dict_output_convet = np.array([-1 if ele == "no" else 1 for ele in dict_output])
        
    return Trees