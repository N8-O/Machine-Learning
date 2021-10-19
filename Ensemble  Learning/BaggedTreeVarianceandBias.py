# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:53:47 2021

@author: Nathan
"""

import time
from BankDataRead import ReadandOrganizeBankData
from ID3_algorithm import *
import numpy as np
import matplotlib.pyplot as plt
from BaggedTrees import BagTrees

t0 = time.time()

data, labels, test_data, test_labels, attribute_classifiers, label_classifiers = ReadandOrganizeBankData() # read in bank data
inf_gain = "H" # information gain method
tree_depth = 17 # tree depths
iterations = 500 # number of trees to make
num_bagged_predictors = 100
weights = 1 # uniform weights
m = 1000
Bagged_predictors = []
Bagged_Test_Err = []

for i in range(num_bagged_predictors):
    passed_data = {}
    passed_labels = []
    resampled_index = np.random.choice(range(len(labels)), 1000, replace=False)
    for key in data:
        passed_data[key] = [data[key][index] for index in resampled_index]
    passed_labels = [labels[index] for index in resampled_index]
    
    Trees = BagTrees(passed_data,passed_labels,test_data,test_labels,attribute_classifiers,label_classifiers,inf_gain,tree_depth,iterations,weights,m)
    Bagged_predictors.append(Trees)

# Total_Train_Err = PassData2BaggedDecisionTrees(Trees,data,labels)
# Total_Test_Err = PassData2BaggedDecisionTrees(Trees,test_data,test_labels)



single_tree_predictions = []
total_tree_predictions = []

for alltrees in Bagged_predictors:
    hypoth_s = BaggedDecisionTreesHypothesis([alltrees[0]],test_data,test_labels)
    hypoth_a = BaggedDecisionTreesHypothesis(alltrees,test_data,test_labels)
    single_tree_predictions.append(hypoth_s[0])
    total_tree_predictions.append(hypoth_a[-1])

labels_converted = np.array([-1 if ele == "no" else 1 for ele in test_labels])


single_tree_predictions = np.array(single_tree_predictions)
bias_s = (single_tree_predictions.mean(axis=0)-labels_converted)**2
mean_bias_s = bias_s.mean()
var_s = single_tree_predictions.var(axis = 0)
mean_var_s = var_s.mean()
single_gen_square_err = mean_bias_s+mean_var_s


total_tree_predictions = np.array(total_tree_predictions)
bias_a = (total_tree_predictions.mean(axis=0)-labels_converted)**2
mean_bias_a = bias_a.mean()
var_a = total_tree_predictions.var(axis = 0)
mean_var_a = var_a.mean()
total_gen_square_err = mean_bias_a+mean_var_a

    
# tree_predictions_average = sum(total_tree_predictions)/num_bagged_predictors

# single_tree_predictions_average[0]


t1 = time.time() # end timer




total = t1-t0  # calc time passsed
print("Code took " + str(total/60) + " minutes to run.")

# plotting figures
# plt.figure(0)
# plt.plot(range(1,iterations+1),Total_Train_Err,'b-')
# plt.plot(range(1,iterations+1),Total_Test_Err,'r-')
# plt.title("Bagged Tree Error As Trees Increases")
# plt.xlim([0,iterations])
# plt.ylim([min(Total_Train_Err)-.001,max([max(Total_Train_Err),max(Total_Test_Err)])+.001])
# plt.xlabel("Number of Trees")
# plt.ylabel("Error")
# plt.legend(["Training Error","Testing Error"])