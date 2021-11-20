# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:50:44 2021

@author: Nathan
"""
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import minimize as mini
from scipy.spatial.distance import pdist, squareform


# function to read in bank note data
def ReadandOrganizeBankNoteData(file):
    data = np.genfromtxt(file,delimiter=',')
    
    data_x = data[:,0:-1]
    # add in bias term
    data_x = np.append(data_x,np.ones((len(data_x),1)),axis=1)
    
    # label data
    data_y = data[:,-1]
    # make 0's in data be -1's
    data_y = np.where(data_y == 0, -1, data_y)
    return data_x, data_y

def shuffle_data(x,labels):
    new_ind = np.random.choice(range(len(labels)),len(labels),replace=False)
    new_x = x[new_ind]
    new_labels = labels[new_ind]
    return new_x, new_labels


def SVM_Primal(xs,labels,epochs,gamma,a,C,schedule):
    w = np.zeros(len(xs[0,:]))
    for epoch in range(epochs):
        # shuffle data
        x_passed, labels_passed = shuffle_data(xs,labels)
        for i in range(len(labels)):
            if schedule == "a":
                gamma_t = gamma/(1+((gamma/a)*i))
            elif schedule =="b":
                gamma_t = gamma/(1+i)
            wx = w.dot(x_passed[i])
            update_test = labels_passed[i]*wx
            if update_test <= 1:
                w = np.array((1-gamma_t)*w)+(gamma_t*C*len(labels_passed)*labels_passed[i]*x_passed[i])
            else:
                w[:-1] = (1-gamma_t)*w[:-1]
    return w
    
def constraint_func(alphas, labels):
    return alphas.dot(labels)

def dual_objective_function(alphas, subgradient):
    result = -sum(alphas)
    result += alphas.dot(subgradient).dot(alphas)
    return 0.5 * result

def gaussian_kernel(xi, xj, gamma):
    result = np.exp(-np.linalg.norm(xi-xj)**2 / gamma)
    return result

def dual_svm(weight_constant, gamma, labels, xs):

    weight = np.zeros(len(xs[0, :-1]))
    stuff = []

    mini_bounds = len(xs)*[(0,weight_constant)]
        
    mini_constraints = {'type': 'eq', 'fun': constraint_func, 'args': (labels,)}

    if gamma is not None:
        pairwise_dists = squareform(pdist(xs, 'euclidean'))
        gram_matrix = np.exp(-pairwise_dists ** 2 / gamma)
        
    else:
        gram_matrix = (xs @ xs.T)

    subgradient = gram_matrix * np.tile(labels.T, (len(gram_matrix), 1)).T * np.tile(labels,(len(gram_matrix), 1))

    objective_result = mini(dual_objective_function, np.zeros(len(xs)), args=subgradient, method='SLSQP', bounds=mini_bounds, constraints=mini_constraints)

    dual_result_alphas = objective_result.x


    if gamma is None:
        weight = (dual_result_alphas * labels).dot(xs)
    else:
        for i in range(len(weight)):
            weight += dual_result_alphas[i] * labels[i] * gaussian_kernel(xs[i],np.zeros(len(xs[i])), gamma)

    # did not finish implementation
    positive_alphas = 0
    for i in range(len(xs)):
        if dual_result_alphas[i] > 0:
            positive_alphas += 1

    b = labels - (dual_result_alphas * labels).dot(gram_matrix)

    b = sum(b)/positive_alphas

    stuff.append(np.insert(weight, len(weight), b))
    # stuff.append(positive_alphas)
    # stuff.append(dual_result)
    return stuff


def PrimalPrediction(xs,w):
    a = w*xs
    out = np.sign(np.sum(a,axis = 1))
    return out

def DualPrediction(xs,w):
    a = w*xs
    out = np.sign(np.sum(a,axis = 1))
    return out

# names of files in to run perceptron algorithm
test_file = "bank-note/test.csv"
training_file = "bank-note/train.csv"

# save data for training and testing
train_x, train_label = ReadandOrganizeBankNoteData(training_file)
test_x, test_label = ReadandOrganizeBankNoteData(test_file)

# run standard perceptron algorithm
epochs = 100
C = np.array([100/873,500/873,700/873])
schedule = "a"
gamma = 1
a = .5


for c in C:
    s_w = SVM_Primal(train_x,train_label,epochs,gamma,a,c,schedule)
    std_train_predictions = PrimalPrediction(train_x,s_w)
    std_test_predictions = PrimalPrediction(test_x,s_w)
    std_train_error = (sum(abs(train_label-std_train_predictions))/2)/len(train_label)
    std_test_error = (sum(abs(test_label-std_test_predictions))/2)/len(test_label)
    print("SVM Primal (a)")
    print("Weights: " + str(s_w))
    print("Average Prediction Error on Training Data: " + str(std_train_error))
    print("Average Prediction Error on Test Data: " + str(std_test_error))
    print("\n")

schedule = "b"
for c in C:
    s_w = SVM_Primal(train_x,train_label,epochs,gamma,a,c,schedule)
    std_train_predictions = PrimalPrediction(train_x,s_w)
    std_test_predictions = PrimalPrediction(test_x,s_w)
    std_train_error = (sum(abs(train_label-std_train_predictions))/2)/len(train_label)
    std_test_error = (sum(abs(test_label-std_test_predictions))/2)/len(test_label)
    print("SVM Primal (b)")
    print("Weights: " + str(s_w))
    print("Average Prediction Error on Training Data: " + str(std_train_error))
    print("Average Prediction Error on Test Data: " + str(std_test_error))
    print("\n")


for c in C:
    dual_w = dual_svm(c,None,train_label, train_x)
    dual_w = dual_w[0][0:-1]
    std_train_predictions = PrimalPrediction(train_x,dual_w)
    std_test_predictions = PrimalPrediction(test_x,dual_w)
    std_train_error = (sum(abs(train_label-std_train_predictions))/2)/len(train_label)
    std_test_error = (sum(abs(test_label-std_test_predictions))/2)/len(test_label)
    print("SVM Dual (non-kernal)")
    print("Weights: " + str(dual_w))
    print("C: " + str(c))
    print("Average Prediction Error on Training Data: " + str(std_train_error))
    print("Average Prediction Error on Test Data: " + str(std_test_error))
    print("\n")
    
    
gammas = [.1,.5,1,5,100]
for gamma in gammas:
    for c in C:
        dual_w = dual_svm(c,gamma,train_label, train_x)
        dual_w = dual_w[0]
        std_train_predictions = DualPrediction(train_x,dual_w)
        std_test_predictions = DualPrediction(test_x,dual_w)
        std_train_error = (sum(abs(train_label-std_train_predictions))/2)/len(train_label)
        std_test_error = (sum(abs(test_label-std_test_predictions))/2)/len(test_label)
        print("SVM Dual (kernal)")
        print("Weights: " + str(dual_w))
        print("gamma: "+ str(gamma))
        print("C: " + str(c))
        print("Average Prediction Error on Training Data: " + str(std_train_error))
        print("Average Prediction Error on Test Data: " + str(std_test_error))
        print("\n")








