# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:50:44 2021

@author: Nathan
"""
import numpy as np
import matplotlib.pyplot as plt

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

# function for standard perceptron algorithm
def StandardPerceptron(xs,labels,epochs,r):
    w = np.zeros((1,len(train_x[0])))
    for epoch in range(epochs):
        # shuffle data
        x_passed, labels_passed = shuffle_data(xs,labels)
        for i in range(len(labels)):
            wx = w*x_passed[i]
            sum_wx = np.sum(wx,axis = 1)
            update_test = labels_passed[i]*sum_wx
            if update_test <= 0:
                w = w+r*labels_passed[i]*x_passed[i]
    return w

def VotedPerceptron(xs,labels,epochs,r):
    w = np.zeros((1,len(train_x[0])))
    m = 0
    c = np.array([0])
    for epoch in range(epochs):
        for i in range(len(labels)):
            wx = w[m]*xs[i]
            sum_wx = np.sum(wx)
            update_test = labels[i]*sum_wx
            if update_test <= 0:
                new_w = (w[m]+(r*labels[i]*xs[i]))
                w = np.append(w,[new_w],axis = 0)
                m = m+1
                c = np.append(c,1)
            else:
                c[m] = c[m]+1
    return w,c

def AveragePerceptron(xs,labels,epochs,r):
    w = np.zeros((1,len(train_x[0])))
    a = np.zeros((1,len(train_x[0])))
    for epoch in range(epochs):
        for i in range(len(labels)):
            wx = w*xs[i]
            sum_wx = np.sum(wx,axis = 1)
            update_test = labels[i]*sum_wx
            if update_test <= 0:
                w = w+r*labels[i]*xs[i]
            a = a+w
    return a
    

def StandardPrediction(xs,w):
    a = w*xs
    out = np.sign(np.sum(a,axis = 1))
    return out

def VotedPrediction(xs,w,c):
    out = 0
    for i in range(len(c)):
        a = w[i]*xs
        a = np.sign(np.sum(a,axis = 1))
        out = out+c[i]*a
    out = np.sign(out)
    return out

def AveragePrediction(xs,w,c):
    out = 0
    for i in range(len(c)):
        a = w[i]*xs
        a = np.sign(np.sum(a,axis = 1))
        out = out+c[i]*a
    out = np.sign(out)
    return out


# names of files in to run perceptron algorithm
test_file = "bank-note/test.csv"
training_file = "bank-note/train.csv"

# save data for training and testing
train_x, train_label = ReadandOrganizeBankNoteData(training_file)
test_x, test_label = ReadandOrganizeBankNoteData(test_file)


# run standard perceptron algorithm
T = 10
r = 0.5
s_w = StandardPerceptron(train_x,train_label,T,r)
std_train_predictions = StandardPrediction(train_x,s_w)
std_test_predictions = StandardPrediction(test_x,s_w)
std_train_error = (sum(abs(train_label-std_train_predictions))/2)/len(train_label)
std_test_error = (sum(abs(test_label-std_test_predictions))/2)/len(test_label)
print("Standard Perceptron")
print("Weights: " + str(s_w))
print("Average Prediction Error on Training Data: " + str(std_train_error))
print("Average Prediction Error on Test Data: " + str(std_test_error))
print("\n")


# run voted perceptron algorithm
T = 10
r = .5
v_w,c = VotedPerceptron(train_x,train_label,T,r)
vt_train_predictions = VotedPrediction(train_x,v_w,c)
vt_test_predictions = VotedPrediction(test_x,v_w,c)
vt_train_error = (sum(abs(train_label-vt_train_predictions))/2)/len(train_label)
vt_test_error = (sum(abs(test_label-vt_test_predictions))/2)/len(test_label)
print("Voted Perceptron")
print("Weights: " + "See CSV file")
export_wc = np.append(v_w,c.reshape(len(c),1),axis=1)
np.savetxt('export_wc.csv',export_wc,delimiter=',')
print("Counts: " + "See CSV file")
print("Average Prediction Error on Training Data: " + str(vt_train_error))
print("Average Prediction Error on Test Data: " + str(vt_test_error))
print("\n")

# run average perceptron algorithm
T = 10
r = 0.5
a = AveragePerceptron(train_x,train_label,T,r)
avg_train_predictions = StandardPrediction(train_x,a)
avg_test_predictions = StandardPrediction(test_x,a)
avg_train_error = (sum(abs(train_label-avg_train_predictions))/2)/len(train_label)
avg_test_error = (sum(abs(test_label-avg_test_predictions))/2)/len(test_label)
print("Average Perceptron")
print("Weights: " + str(a))
print("Average Prediction Error on Training Data: " + str(avg_train_error))
print("Average Prediction Error on Test Data: " + str(avg_test_error))








