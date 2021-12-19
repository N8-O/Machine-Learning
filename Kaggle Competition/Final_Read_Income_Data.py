#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:39:45 2021

@author: nathan
"""
# import numpy as np # linear algebra
import pandas as pd # data processing
# import matplotlib.pyplot as plt
# import csv
# import seaborn as sns

# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import scale, MinMaxScaler
# from sklearn import svm



# from tensorflow import keras
# import tensorflow as tf
# from keras.wrappers.scikit_learn import KerasClassifier


def readIncomeData(train_file,test_file):
    datafile = pd.read_csv(train_file, sep=",")
    datafile.head()
    
    testdata = pd.read_csv(test_file,sep=",")
    testdata.head()
    testdata = testdata.drop('ID', axis=1)
    
    # Convert data to integars
    # datafile['native.country'] = datafile['native.country'].replace('United-States','0')
    # datafile['native.country']= datafile['native.country'].where((df['native.country']=='0') | (datafile['native.country']=='2'),'1')
    # datafile['native.country']=datafile['native.country'].map({'0': 0, '1': 1, '2':2})
    
    # datafile['workclass']=datafile['workclass'].map({'Self-emp-not-inc': 0, '?': 1, 'Private':2, 'Local-gov':3, 'State-gov':4,'Self-emp-inc':5,'Federal-gov':6,'Without-pay':7,'Never-worked':8})
    # datafile['marital.status']=datafile['marital.status'].map({'Married-civ-spouse': 0, 'Never-married': 1, 'Divorced':2, 'Separated':3, 'Widowed':4,'Married-spouse-absent':5,'Married-AF-spouse':6})
    # datafile['occupation']=datafile['occupation'].map({'Prof-specialty': 0, 'Craft-repair': 1, 
    #                 'Exec-managerial':2, 'Adm-clerical':3, 'Sales':4,'Other-service':5, 
    #                 'Machine-op-inspct':6, '?':7,'Transport-moving':8,'Handlers-cleaners':9,
    #                 'Farming-fishing':10,'Tech-support':11,'Protective-serv':12,'Priv-house-serv':13,
    #                 'Armed-Forces':14})
    # datafile['relationship'] = datafile['relationship'].map({'Husband': 0, 'Not-in-family': 1, 
    #                 'Own-child':2, 'Unmarried':3, 'Wife':4,'Other-relative':5})
    # datafile['race'] = datafile['race'].map({'White': 0, 'Black': 1, 
    #                 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
    # datafile['sex'] = datafile['sex'].map({'Male': 0, 'Female': 1})
    
    # testdata['native.country'] = testdata['native.country'].replace('United-States','0')
    # testdata['native.country']= testdata['native.country'].where((testdata['native.country']=='0') | (testdata['native.country']=='2'),'1')
    # testdata['native.country']=testdata['native.country'].map({'0': 0, '1': 1, '2':2})
    
    # testdata['workclass'] = testdata['workclass'].map({'Self-emp-not-inc': 0, '?': 1, 'Private':2, 'Local-gov':3, 'State-gov':4,'Self-emp-inc':5,'Federal-gov':6,'Without-pay':7,'Never-worked':8})
    # testdata['marital.status'] = testdata['marital.status'].map({'Married-civ-spouse': 0, 'Never-married': 1, 'Divorced':2, 'Separated':3, 'Widowed':4,'Married-spouse-absent':5,'Married-AF-spouse':6})
    # testdata['occupation'] = testdata['occupation'].map({'Prof-specialty': 0, 'Craft-repair': 1, 
    #                 'Exec-managerial':2, 'Adm-clerical':3, 'Sales':4,'Other-service':5, 
    #                 'Machine-op-inspct':6, '?':7,'Transport-moving':8,'Handlers-cleaners':9,
    #                 'Farming-fishing':10,'Tech-support':11,'Protective-serv':12,'Priv-house-serv':13,
    #                 'Armed-Forces':14})
    # testdata['relationship'] = testdata['relationship'].map({'Husband': 0, 'Not-in-family': 1, 
    #                 'Own-child':2, 'Unmarried':3, 'Wife':4,'Other-relative':5})
    # testdata['race'] = testdata['race'].map({'White': 0, 'Black': 1, 
    #                 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
    # testdata['sex'] = testdata['sex'].map({'Male': 0, 'Female': 1})
    
    
    # Correlation matrix
    # plt.figure(figsize=(15, 8))
    # plt.title('Correlation of Variables and Income')
    # sns.heatmap(abs(datafile.corr()), annot=True, vmin=0, vmax=.5,cmap="YlGnBu")
    
    
    # Drop useless data
    datafile = datafile.drop('native.country', axis=1)
    testdata = testdata.drop('native.country', axis=1)
    
    datafile = datafile.drop('race', axis=1)
    testdata = testdata.drop('race', axis=1)
    
    datafile = datafile.drop('fnlwgt', axis=1)
    testdata = testdata.drop('fnlwgt', axis=1)
    
    datafile = datafile.drop('education', axis = 1)
    testdata = testdata.drop('education', axis=1)
    
    X = datafile.drop('income>50K',axis=1)
    Y = datafile['income>50K']
    Y = Y.replace(0,-1)
    
    X_categorical = X[['workclass', 'marital.status', 'occupation', 'relationship',
                        'sex']]
    X_continous  = X[['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num']]
    X_encoded = pd.get_dummies(X_categorical)
    X = pd.concat([X_continous, X_encoded],axis=1)
    traindata = pd.concat([X,Y],axis=1)
    
    
    testdata_categorical = testdata[['workclass', 'marital.status', 'occupation', 'relationship',
                        'sex']]
    testdata_continous  = testdata[['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num']]
    testdata_encoded = pd.get_dummies(testdata_categorical)
    testdata = pd.concat([testdata_continous, testdata_encoded],axis=1)
    
    return traindata,testdata

# get data
# train_file = "income_data/train_final.csv"
# test_file = "income_data/test_final.csv"
# train_x, train_y, test_x = readIncomeData(train_file, test_file)

# # Prepare the data
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 1/3, stratify=Y,random_state=10 )




# # MODEL - LOGISTIC REGRESSION
# logit = LogisticRegression(tol=0.001, max_iter=100000)
# logit = logit.fit(X_train, y_train)

# # CROSS VALIDATION
# cv = StratifiedKFold(n_splits=3) # we make 3 splits
# val_logit = cross_val_score(logit, X_train, y_train, cv=cv).mean()
# print(val_logit) # show validation set score

# logit_predictions = logit.predict(X_test)
# acc_logit = accuracy_score(y_test,logit_predictions)
# print(acc_logit) # show test set score

# testdata_predict = logit.predict(testdata)




# MODEL - SVM
# X_train = scale(X_train)
# X_test = scale(X_test)

# suppvm = svm.SVC(kernel='linear')
# suppvm = suppvm.fit(X_train, y_train)

# # CROSS VALIDATION
# cv = StratifiedKFold(n_splits=3)
# val_suppvm = cross_val_score(suppvm, X_train, y_train, cv=cv).mean()
# print(val_suppvm)

# suppvm_predictions = suppvm.predict(X_test)
# acc_suppvm = accuracy_score(y_test,suppvm_predictions)
# print(acc_suppvm)

# testdata = scale(testdata)
# testdata_predict = suppvm.predict(testdata)









# head = ["ID", "Prediction"]

# with open('Nathan_result_file.csv', 'w+', newline='') as csvfile:
#   writer = csv.writer(csvfile, dialect='excel')
#   writer.writerow(head)
#   ID = 1
#   for l in testdata_predict:
#     writer.writerow([ID,l])
#     ID = ID+1