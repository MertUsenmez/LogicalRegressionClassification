# -*- coding utf-8 -*-


#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Description Dataset

# Our data for prediction about Benign Cancer and Malignant Cancer
data = pd.read_csv("data.csv")
print(data.info)

# Columns id and Unnamed: 32 have no effect in our classification, so we drop them
data.drop(["Unnamed: 32", "id"], axis=1, inplace = True)

# You can check the diagnosis feature with the object type (print(data.info))
# This feature must be either cathegorical or integer
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]
# diagnosis be integer.
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

#%% Normalization
# I have to normalize all of my features because a feature provide that i can ignore another feature
# The reason for this is a value difference between the value of a feature and the value of the property that can be ignored, although they are all float.
# For this I will scale all the featurda between 0 and 1
# Normalization Formula = (x-min(x)) / (max(x)-min(x))
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#%% Train Test Split
from sklearn.model_selection import train_test_split

# test size 20%, so, 20% of data divide for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

#%% parameter initialize and sigmoid function

def initialize_weight_and_bias(dimension):

    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

# dimension = 30
weight,bias = initialize_weight_and_bias(30)

# Sigmoid function formula: f(x)=1/(1+e^-(x))
def sigmoid(z):
    
    y_head = 1/(1 + np.exp(-z))
    return y_head

print(sigmoid(6))

#%%

def forward_backward_propagation(weight,bias,x_train,y_train):
    # forward propagation
    # The reason why we get transposed weight because (30,1) * (30,455) cannot multiply the matrices, we are converting it to (1,30) * (30,455)
    z = np.dot(weight.T, x_train) + bias
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # x_train.shape[1] for scaling
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward propagation
    # x_train.shape[1] for scaling
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    # x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


#%% Updating(learning) (TRAIN) parameters with FOR LOOP

def update(weight, bias, x_train, y_train, learning_rate, number_of_iteration):

    cost_list = []   # I'll use it to store all costs
    cost_list2 = []  # I'll use it to store the costs that occur in every 10 steps
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(weight, bias, x_train, y_train)
        cost_list.append(cost)
        
        #lets update
        weight = weight - learning_rate * gradients["derivative_weight"]
        bias = bias - learning_rate * gradients["derivative_bias"]
        
        if i%10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight":weight, "bias":bias}
    plt.plot(index,cost_list2)
    plt.xticks(index, rotation="vertical")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list 


#%% PREDICTION so forward propagation
def predict(weight, bias, x_test):
    # x_test is my normalized dataset ie input for forward propagation
    z = sigmoid(np.dot(weight.T, x_test) + bias)
    Y_prediction = np.zeros(1,x_test.shape[1])
    
    # if z is bigger than 0.5, our prediction is sign one (y_head = 1)
    # if z is smaller than 0.5, our prediction is sign zero (y_head = 0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction

#%% logistic_regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_iterations):
    
    dimension = x_train.shape[0] # is 4096
    weight,bias = initialize_weight_and_bias(dimension)
    
    parameters, gradients, cost_list = update(weight, bias, x_train, y_train, learning_rate, number_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    
    print("Train Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
# learning_rate and number_iterations parameters are hiper-parameters. These values should be changed manually when they are needed
# Prediction more successful or more unsuccessful
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, number_iterations = 3000)




#%% Logistic Regression with sklearn  -----------------------------------

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print("Test Accuracy {}".format(lr.score(x_test.T, y_test.T)))
            
            
            
            








