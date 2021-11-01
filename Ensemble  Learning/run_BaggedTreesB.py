# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:03:14 2021

@author: Nathan
"""
import time
from BankDataRead import ReadandOrganizeBankData
from ID3_algorithm import PassData2BaggedDecisionTrees
import matplotlib.pyplot as plt
from BaggedTrees import BagTrees
# start timer
t0 = time.time()

data, labels, test_data, test_labels, attribute_classifiers, label_classifiers = ReadandOrganizeBankData() # read in bank data
inf_gain = "H" # information gain method
tree_depth = 17 # tree depths
iterations = 500 # number of trees to make
weights = 1 # uniform weights
m = 'rand'
Trees = BagTrees(data,labels,test_data,test_labels,attribute_classifiers,label_classifiers,inf_gain,tree_depth,iterations,weights,m)

Total_Train_Err = PassData2BaggedDecisionTrees(Trees,data,labels)
Total_Test_Err = PassData2BaggedDecisionTrees(Trees,test_data,test_labels)

t1 = time.time() # end timer
total = t1-t0  # calc time passsed
print("Code took " + str(total/60) + " minutes to run.")

# plotting figures
plt.figure(0)
plt.plot(range(1,iterations+1),Total_Train_Err,'b-')
plt.plot(range(1,iterations+1),Total_Test_Err,'r-')
plt.title("Bagged Tree Error As Trees Increases")
plt.xlim([0,iterations])
plt.ylim([min(Total_Train_Err)-.001,max([max(Total_Train_Err),max(Total_Test_Err)])+.001])
plt.xlabel("Number of Trees")
plt.ylabel("Error")
plt.legend(["Training Error","Testing Error"])
plt.show()
