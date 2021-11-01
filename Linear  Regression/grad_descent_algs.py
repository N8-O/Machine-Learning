# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:33:20 2021

@author: Nathan
"""
import numpy as np
import matplotlib.pyplot as plt

# gradient decent

def gradient_decent(y,x,w,r,tolerance=1e-06,max_steps = 50000):
    d=np.inf
    i = 0
    cost = np.array([])
    while d>tolerance and i<max_steps:
        err = np.zeros(np.shape(y))
        wx = np.matmul(w[i],x)
        err = y-wx
        cost = np.append(cost,.5*np.sum(err**2))
        grad = -np.sum((err*x),axis=1)
        new_w = w[i]-r*grad
        w = np.append(w,[new_w],axis=0)
        if len(w)>1:
            d = np.linalg.norm(w[i]-w[i-1])
        i = i+1
    return w,cost
    
def gradient_decent_test(y,x,w):
    err = np.zeros(np.shape(y))
    wx = np.matmul(w,x)
    err = y-wx
    cost = .5*np.sum(err**2)
    return cost
        
        # w = (XX^T)^-1 XY
        # np.matmul(np.matmul(np.linalg.inv(np.matmul(x,x.T)),x),y)
    
        
def stochastic_gradient_decent(y,x,w,steps,r,test_x,test_y):
    i=0
    # cost = []
    sgd_test_cost=[]
    while i<steps:
        randx = np.random.randint(0,len(y))
        test_ex = x[:,randx]
        err = np.zeros(np.shape(y))
        wx = sum(w[i]*test_ex)
        # print("wx: " +str(wx))
        err = y[randx]-wx
        # print("error: "+ str(err))
        # new_cost = .5*np.sum(err**2)
        
        # cost.append(new_cost)
        
        stoch_grad = err*test_ex
        # print("grad: " + str(stoch_grad))
        new_w = w[i] + r*stoch_grad
        # print("new w: " + str(new_w))
        w = np.append(w,[new_w],axis=0)
        i = i+1
        sgd_test_cost.append(gradient_decent_test(test_y,test_x,new_w))
    return w,sgd_test_cost

# read in file
def ReadandOrganizeConcreteData(file):
    data = np.genfromtxt(file,delimiter=',')
    
    data_x = data[:,0:-1]
    data_y = data[:,-1]

    return data_x, data_y


# steps = 1000
# r = 0.05
# y = np.array([1,4,-1,-2,0])
# x = np.array([[1,1,1,1,1],[1,1,-1,1,3],[-1,1,1,2,-1],[2,3,0,-4,-1]])
# w = np.array([[-1,-1,1,-1]])

# ans = gradient_decent(y,x,w,steps,r)

# steps = 100
# r = 0.1
# y = np.array([1,4,-1,-2,0])
# x = np.array([[1,1,1,1,1],[1,1,-1,1,3],[-1,1,1,2,-1],[2,3,0,-4,-1]])
# w = np.array([[0,0,0,0]])

# ans2 = stochastic_gradient_decent(y,x,w,steps,r)

# ans3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x,x.T)),x),y)

steps = 1000
r = 0.05

train_file = "concrete/train.csv"
test_file = "concrete/test.csv"

train_x, train_y = ReadandOrganizeConcreteData(train_file)
test_x, test_y = ReadandOrganizeConcreteData(test_file)

# insert bias ones!
train_x = np.insert(train_x,0,1,axis=1)
test_x = np.insert(test_x,0,1,axis=1)


w = np.zeros((1,len(train_x[0])))

train_x = train_x.T
test_x = test_x.T


opt = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x,train_x.T)),train_x),train_y)

# steps = 1000
r =  .015
gd_ans,gd_train_cost = gradient_decent(train_y,train_x,w,r)
gd_test_cost = gradient_decent_test(test_y,test_x,gd_ans[-1])

w = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

steps = 20000
r = .0001
sgd_ans,sgd_train_cost = stochastic_gradient_decent(train_y,train_x,w,steps,r,test_x,test_y)
# sgd_test_cost = gradient_decent_test(test_y,test_x,sgd_ans[-1])

plt.figure()
plt.plot(range(len(gd_train_cost)),gd_train_cost)
plt.title("Gradient Decent Cost Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

plt.figure()
plt.plot(range(len(sgd_train_cost)),sgd_train_cost)
plt.title("Stochastic Gradient Decent Cost Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()









