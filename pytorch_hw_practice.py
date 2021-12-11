#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:31:15 2021

@author: nathan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime



class Net(nn.Module):

    def __init__(self, input_channels, depth, width, typ):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.depth = depth
        self.input_channels = input_channels
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        
        self.fc1 = nn.Linear(input_channels,width)
        self.fc2 = nn.Linear(width,width)
        self.fc3 = nn.Linear(width,width)
        self.fc4 = nn.Linear(width,width)
        self.fc5 = nn.Linear(width,width)
        self.fc6 = nn.Linear(width,width)
        self.fc7 = nn.Linear(width,width)
        self.fc8 = nn.Linear(width,width)
        self.fc9 = nn.Linear(width,1)
        
        if typ == 'tanh':
                torch.nn.init.xavier_uniform_(self.fc1.weight)
                torch.nn.init.xavier_uniform_(self.fc2.weight)
                torch.nn.init.xavier_uniform_(self.fc3.weight)
                torch.nn.init.xavier_uniform_(self.fc4.weight)
                torch.nn.init.xavier_uniform_(self.fc5.weight)
                torch.nn.init.xavier_uniform_(self.fc6.weight)
                torch.nn.init.xavier_uniform_(self.fc7.weight)
                torch.nn.init.xavier_uniform_(self.fc8.weight)
                torch.nn.init.xavier_uniform_(self.fc9.weight)
                
        if typ == 'RELU':
                torch.nn.init.kaiming_uniform_(self.fc1.weight)
                torch.nn.init.kaiming_uniform_(self.fc2.weight)
                torch.nn.init.kaiming_uniform_(self.fc3.weight)
                torch.nn.init.kaiming_uniform_(self.fc4.weight)
                torch.nn.init.kaiming_uniform_(self.fc5.weight)
                torch.nn.init.kaiming_uniform_(self.fc6.weight)
                torch.nn.init.kaiming_uniform_(self.fc7.weight)
                torch.nn.init.kaiming_uniform_(self.fc8.weight)
                torch.nn.init.kaiming_uniform_(self.fc9.weight)

    def forward(self, x, depth, typ):
        if depth == 3:
            if typ == 'tanh':
                x = F.tanh(self.fc1(x))
                x = F.tanh(self.fc2(x))
                x = self.fc9(x)
                return x
            if typ == 'RELU':
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc9(x)
                return x
        elif depth == 5:
            if typ == 'tanh':
                x = F.tanh(self.fc1(x))
                x = F.tanh(self.fc2(x))
                x = F.tanh(self.fc3(x))
                x = F.tanh(self.fc4(x))
                x = self.fc9(x)
                return x
            if typ == 'RELU':
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x =self.fc9(x)
                return x
        elif depth == 9:
            if typ == 'tanh':              
                x = F.tanh(self.fc1(x))
                x = F.tanh(self.fc2(x))
                x = F.tanh(self.fc3(x))
                x = F.tanh(self.fc4(x))
                x = F.tanh(self.fc5(x))
                x = F.tanh(self.fc6(x))
                x = F.tanh(self.fc7(x))
                x = F.tanh(self.fc8(x))
                x =self.fc9(x)
                return x
            if typ == 'RELU':                
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                x = F.relu(self.fc8(x))
                x = self.fc9(x)
                return x
        

def train_one_epoch(epoch_index, model, inputs, labels,depth,typ):
    running_loss = 0.
    last_loss = 0.
    other_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    r = 0
    for inp in inputs:
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # inp = torch.Tensor(inp)
        # Make predictions for this batch
        
        outputs = model(inp,depth,typ)#.unsqueeze(dim=0)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels[r].unsqueeze(dim=0))
        
        if outputs>.5 and labels[r].unsqueeze(dim=0)>.5:
            other_loss = other_loss+1
        if outputs<.5 and labels[r].unsqueeze(dim=0)<.5:
            other_loss = other_loss+1
        # print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # print(running_loss)
        if r % len(labels) == 871:
            last_loss = running_loss / len(labels) # loss per batch
            other_loss = other_loss/len(labels)
            
            # print('  batch {} loss: {}'.format(r + 1, last_loss))
            # tb_x = epoch_index * len(labels) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        r = r+1
    return last_loss,other_loss

# function to read in bank note data
def ReadandOrganizeBankNoteData(file):
    data = np.genfromtxt(file,delimiter=',')
    
    data_x = data[:,0:-1]
    # add in bias term
    data_x = np.append(data_x,np.ones((len(data_x),1)),axis=1)
    
    # label data
    data_y = data[:,-1]
    # make 0's in data be -1's
    # data_y = np.where(data_y == 0, -1, data_y)
    return data_x, data_y

def shuffle_data(x,labels):
    new_ind = np.random.choice(range(len(labels)),len(labels),replace=False)
    new_x = x[new_ind]
    new_labels = labels[new_ind]
    return new_x, new_labels

# names of files in to run perceptron algorithm
test_file = "bank-note/test.csv"
training_file = "bank-note/train.csv"

# save data for training and testing
train_x, train_label = ReadandOrganizeBankNoteData(training_file)
test_x, test_label = ReadandOrganizeBankNoteData(test_file)
train_x, train_label = shuffle_data(train_x,train_label)

train_x = torch.Tensor(train_x)
train_label = torch.Tensor(train_label)
test_x = torch.Tensor(test_x)
test_label = torch.Tensor(test_label)

NeuralNets = []
i = 0
input_channels = train_x.shape[1]

for typ in ['RELU','tanh']:
    print("Activation Function: " + typ)
    for depth in [3,5,9]:
        print("Depth: "+ str(depth))
        for width in [5,10,25,50,100]:
            print("Width: " + str(width))
            NeuralNets.append(Net(input_channels, depth, width, typ))
            # print(NeuralNets[i])
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(NeuralNets[i].parameters())
            
    
            # Initializing in a separate cell so we can easily add more epochs to the same run
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
            epoch_number = 0
            
            EPOCHS = 1
            
            best_vloss = 1_000_000.
            
            for epoch in range(EPOCHS):
                # print('EPOCH {}:'.format(epoch_number + 1))
                train_x, train_label = shuffle_data(train_x,train_label)
                # Make sure gradient tracking is on, and do a pass over the data
                NeuralNets[i].train(True)
                avg_loss,accur = train_one_epoch(epoch, NeuralNets[i], train_x, train_label,depth,typ)
            
                # We don't need gradients on to do reporting
                NeuralNets[i].train(False)
            
                running_vloss = 0.0
                other_loss = 0
                r = 0
                for tes in test_x:
                    voutputs = NeuralNets[i](tes,depth,typ)
                    vloss = loss_fn(voutputs, test_label[r].unsqueeze(dim=0))
                    running_vloss += vloss
                    if voutputs>.5 and test_label[r].unsqueeze(dim=0)>.5:
                        other_loss = other_loss+1
                    if voutputs<.5 and test_label[r].unsqueeze(dim=0)<.5:
                        other_loss = other_loss+1
                    r = r+1
                
                avg_vloss = running_vloss / (epoch + 1)
                # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            
                if epoch == EPOCHS-1:
                    print("Training Error: "+str(accur))
                    print("Testing Error: "+str(other_loss/len(test_label)))
                    
                # Log the running loss averaged per batch
                # for both training and validation
                # writer.add_scalars('Training vs. Validation Loss',
                                # { 'Training' : avg_loss, 'Validation' : avg_vloss },
                                # epoch_number + 1)
                # writer.flush()
            
                # Track best performance, and save the model's state
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                    # torch.save(NeuralNets[i].state_dict(), model_path)
            
                epoch_number += 1
            
            
            
            
            i = i+1