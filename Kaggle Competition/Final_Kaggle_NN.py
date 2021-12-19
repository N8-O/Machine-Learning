#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:23:59 2021

@author: nathan
"""
from Final_Read_Income_Data import readIncomeData
import numpy as np
from sklearn.preprocessing import scale
import csv

import torch
import torch.nn as nn

# BASE NEURAL NET CLASS
class Net(nn.Module):
    def __init__(self,layers,act=nn.Tanh()):
        super(Net,self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers)-1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
            
    def forward(self,x):
        for i in range(len(self.fc)-1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x

def shuffle_data(x,batch_sz):
    negs = np.where(train_data[:,-1]<0)[0]   
    new_ind = np.random.choice(negs,int(batch_sz/2),replace=False)
    new_x_negs = x[new_ind]
    posi = np.where(train_data[:,-1]>0)[0]  
    new_ind = np.random.choice(posi,int(batch_sz/2),replace=False)
    new_x_posi = x[new_ind]
    
    new_x = np.append(new_x_negs,new_x_posi,axis=0)
    
    new_ind = np.random.choice(range(len(new_x)),len(new_x),replace=False)
    new_x = x[new_ind]
    return new_x


# get data
train_file = "income_data/train_final.csv"
test_file = "income_data/test_final.csv"
train_data, test_x = readIncomeData(train_file, test_file)
train_data = np.array(train_data)
# train_x = train_data[:,0:-1]
all_train_data = scale(train_data)
new_ind_train_data = np.random.choice(range(len(all_train_data)),len(all_train_data),replace=False)
all_train_data = all_train_data[new_ind_train_data]

train_test_data = all_train_data[0:6250,:]

train_data = all_train_data[6250::,:]

test_x = scale(test_x)

train_data = torch.tensor(train_data)
train_data = torch.tensor(all_train_data)


train_test_data = torch.tensor(train_test_data)
# train_y = train_data[:,-1]
# train_y = torch.tensor(train_y.values)
test_x = torch.Tensor(test_x)

NeuralNets = []
input_channels = [train_data.shape[1]-1]
output_channels = [1]

epochs=[21]
learning_rates=[.008]
reg=1e-6
batch_sz = [6700]
test_num = 1
train_error_ave = []
test_error_ave = []
# train_error = np.array([])#np.zeros((test_num,epochs))
# test_error = np.array([])#np.zeros((test_num,epochs))
for epoch in epochs:
    print('epoch: ' +str(epoch))
    for typ in [nn.Tanh()]:
        print("Activation Function: " + str(typ))
        for depth in [5]:
            print("Depth: "+ str(depth))
            for width in [110]:
                print("Width: " + str(width))
                for lr in learning_rates:
                    w = 0
                    for b_sz in batch_sz:
                        for test in range(test_num):
                            train_error = np.array([])
                            test_error = np.array([])
                            hidden = [width]*depth
                            layers = input_channels+hidden+output_channels
                            NeuralNets.append(Net(layers, typ))
                            # print(NeuralNets[i])
                            optimizer = torch.optim.Adam(NeuralNets[-1].parameters(),lr=lr, weight_decay=reg)
                            loss_fn = nn.BCEWithLogitsLoss()
                            model = NeuralNets[-1]
                            model.train(True)
                            i = 0
                            ii = b_sz
                            
                            for r in range(epoch): 
                                batch_train_data = shuffle_data(train_data,b_sz)
                                
                                
                                optimizer.zero_grad()
                                    
                                x = batch_train_data[:,0:-1].float()
                                x.requires_grad = True
                                y = batch_train_data[:,-1].float().unsqueeze(1)
                                # x.requires_grad=True
                                # print(x.requires_grad)
                                # y = y.float().unsqueeze(1)
                                pred = model(x)
                                # pred = nn.functional.normalize(pred)
                                # pred = pred / pred.amax(keepdim=True)
                                    
                                loss = loss_fn(pred, y)
                                # print(loss.item())
                                loss.backward()
                                optimizer.step()
                
                                if r == epoch-1:
                                    # print(str(r))
                                    
                                    train_results = model(train_data[:,0:-1].float())
                                    a = sum((train_results.squeeze()>=0)*(train_data[:,-1]>=0))
                                    b = sum((train_results.squeeze()<0)*(train_data[:,-1]<0))
                                    acc_tr = (a+b)/len(train_results)
                                    
                                    
                                    # train_test_results = model(train_test_data[:,0:-1].float())
                                    # a = sum((train_test_results.squeeze()>=0)*(train_test_data[:,-1]>=0))
                                    # b = sum((train_test_results.squeeze()<0)*(train_test_data[:,-1]<0))
                                    # acc_tr_test = (a+b)/len(train_test_results)
                                    
                                    train_error=np.append(train_error,acc_tr)
                                    # test_error=np.append(test_error,acc_tr_test)
                                    print(train_error)
                                    # print(test_error)
                                    
                                    
                                    # train_error[test,r] = acc_tr
                                    # test_error[test,r] = acc_tr_test
                                    # print('r', 'Epoch #{}\t: '.format(ie), end='')
                                    # print('bce={:.5f}, train_acc={:.4f}'.format(loss.item(), acc_tr))
                                    # print('train_acc={:.4f}'.format(acc_tr_test))
                                    
                                    
                                    # if (test == test_num-1):
                                    #     train_error_ave.append(np.mean(train_error))
                                    #     test_error_ave.append(np.mean(test_error))
                                    #     print(train_error_ave[-1])
                                    #     print(test_error_ave[-1])

                                i = ii
                                ii = ii+b_sz
                            w = w+1


testdata_predict = model(test_x.float())
testdata_predict = testdata_predict.detach().numpy()
head = ["ID", "Prediction"]

with open('Nathan_result_file.csv', 'w+', newline='') as csvfile:
  writer = csv.writer(csvfile, dialect='excel')
  writer.writerow(head)
  ID = 1
  for l in testdata_predict:
    writer.writerow([ID,l[0]])
    ID = ID+1

                