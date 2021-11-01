# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:53:30 2021

@author: Nathan
"""

# Adaboost!
import numpy as np
from ID3_algorithm import *
from BankDataRead import ReadandOrganizeBankData
import time
import matplotlib.pyplot as plt

def AdaBoost(data, attribute_classifiers, labels, label_classifiers, tree_depth, inf_gain, iterations):
    # initialize importance of all training data
    
    D = []
    D.append([1/len(labels)]*len(labels))
    Train_Err = np.zeros((iterations))
    Train_Err2 = np.zeros((iterations))

    Test_Err = np.zeros((iterations))
    
    Total_Train_Err = np.zeros((iterations))
    Total_Test_Err = np.zeros((iterations))

    H = []
    
    # convert labels into -1 and +1
    labels_convert = np.array([-1 if ele == "no" else 1 for ele in labels])
    
    alpha = np.array([])

    
    for t in range(iterations):
        
        # Find classifier ht whose weighted classification error is better than chance
        H.append(ID3(data, attribute_classifiers, labels, label_classifiers, tree_depth, D[t], inf_gain))
        # calculate error
        Train_Err[t] = PassDataToTestwithWeight(H[t],data,labels,D[t]) 
        Test_Err[t] = PassDataToTest(H[t],test_data,test_labels)
        Train_Err2[t] = PassDataToTest(H[t],data,labels)

        # Compute its vote
        alpha = np.append(alpha,.5*np.log((1-Train_Err[t])/Train_Err[t]))
        
        # Update the values of the weights for the training examples
        new_D = np.array(D[t])
        dict_output = ReturnDictionaryOutput(H[t],data,labels)
        dict_output_convet = np.array([-1 if ele == "no" else 1 for ele in dict_output])
        new_D = (new_D)*np.exp(-alpha[t]*labels_convert*dict_output_convet)
        new_D = new_D/sum(new_D)
        new_D = new_D.tolist()
        
        D.append(new_D)
        
    # Check Error of Training and Test Data on the whole set thus far
    # returns the total training error for each number of iterations
    Total_Train_Err = PassData2Adaboost(alpha,H,data,labels)
    Total_Test_Err = PassData2Adaboost(alpha,H,test_data,test_labels)

    return alpha, H, Train_Err2, Test_Err, Total_Train_Err, Total_Test_Err


# start timer
t0 = time.time()

data, labels, test_data, test_labels, attribute_classifiers, label_classifiers = ReadandOrganizeBankData() # read in bank data
inf_gain = "H" # information gain method
tree_depth = 1 # tree depths
iterations = 500 # number of trees to make
alpha, H, train_err, test_err, Total_Train_Err, Total_Test_Err = AdaBoost(data, attribute_classifiers, labels, label_classifiers, tree_depth, inf_gain, iterations)

t1 = time.time() # end timer
total = t1-t0  # calc time passsed

print("Code took " + str(total/60) + " minutes to run.")

# plotting figures
plt.figure()
plt.plot(range(1,iterations+1),Total_Train_Err,'b-')
plt.plot(range(1,iterations+1),Total_Test_Err,'r-')
plt.title("Error As Number of Stumps Increases")
plt.xlim([1,iterations])
plt.ylim([min(Total_Train_Err)-.001,max([max(Total_Train_Err),max(Total_Test_Err)])+.001])
plt.xlabel("Number of Stumps")
plt.ylabel("Error")
plt.legend(["Training Error","Testing Error"])
plt.show()

plt.figure()
plt.plot(range(1,iterations+1),train_err,'b.')
plt.plot(range(1,iterations+1),test_err,'r.')
plt.title("Error for Each Stump")
plt.xlim([1,iterations])
plt.ylim([min(train_err)-.001,max([max(train_err),max(test_err)])+.001])
plt.xlabel("Stump Number")
plt.ylabel("Error")
plt.legend(["Weighted Training Error","Testing Error"])
plt.show()

    
