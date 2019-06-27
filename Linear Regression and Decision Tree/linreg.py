'''
Aditya Dendukuri HW1

Linear Regression
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.add_subplot(111)   

ax = fig.add_subplot(111)


lmbd = 1.0   #quadratic term
eps =  0.1  #learning criteria
alpha =  0.01   #learning rate

def normalise_vector(vect):
      mag = 0
      for i in range(len(vect)):
            mag += vect[i]*vect[i]
      mag = np.sqrt(mag)
      return vect/mag

#load training data
def load_data(path):
      features = []
      labels = []
      metadata = []
      file_to_parse = open(path, "r")
      lines = file_to_parse.readlines()  
      iter = 0
      a = 0
      for x in lines:
            data = x.split()
            if iter < 19:
                  metadata.append(x)
            else:
                  a+=1
                  features.append(np.array(data[1:16]))
                  labels.append(np.array(data[16]))
            iter+=1
      return features, labels


#convert read data to float
def convert_to_float(features, labels):
      a = []
      b = []
      for i in range(len(features)):
            b.append(float(labels[i]))
            c = []
            for j in range(len(features[i])):
                  c.append(np.array(float(features[i][j])))
            a.append(c)
      return np.array(a), np.array(b)


def feature_normalization(features):
      max = np.amax(features, axis=0)
      min = np.amin(features, axis=0)
      averages = np.mean(features, axis = 0)
      for i in range(len(features)):
            for j in range(len(features[i])):      
                  features[i][j] -= averages[j]
      for i in range(len(features)):
            for j in range(len(features[i])):
                  features[i][j] /= (max[j] - min[j])
      return features


#load features, lables and initialize all variables 
features_s, labels_s = load_data("data.txt")
features, labels = convert_to_float(features_s, labels_s)
features = feature_normalization(features)
labels = normalise_vector(labels)
num_data = len(features)
dim = len(features[0])
print(features_s[0])

# parameters
theta = np.random.randn(dim)

#linear regression model
def model(x, y):
      return np.dot(x, y)
      
#loss function with quadratic regularization
def lossq(x):
      a = 0.0
      b = 0.0
      z = float(1.0/(2.0*float(num_data)))
      for i in range(0, num_data):
            a += (labels[i] - model(features[i], x))**2
      for i in range(0, dim):
            b += (x[i])**2
      return z * (a + lmbd*b)

def loss(x):
      a = 0.0
      z = float(1.0/(2.0*float(num_data)))
      for i in range(0, num_data):
            a += (labels[i] - model(features[i], x))**2
      return z * a 


#gradient of the quadratic loss function 
def lossq_grad(x):
      a = np.zeros(dim)
      b = float(1.0/float(num_data))
      for i in range(0, num_data):
            t1 = float(model(features[i], x)-labels[i])
            a += t1*features[i]
      return b * (a + lmbd*x)

def losslaz(x):
      a = 0.0
      b = 0.0
      z = float(1.0/2*float(num_data))
      for i in range(0, num_data):
            a += (labels[i] - model(features[i], x))**2
      for i in range(0, dim):
            b += abs(x[i])
      return z * (a + lmbd*b)

def losslaz_grad(x):
      a = np.zeros(dim)
      b = float(1.0/2*float(num_data))
      for i in range(0, num_data):
            t1 = 2*float(model(features[i], x)-labels[i])
            a += t1*features[i]
      lmbd_mat = lmbd*np.ones(x.shape)
      return b * (a + lmbd_mat)

def linear_regression(x):
      k = 0
      percent_cost = 1.0
      losses = []
      sq_losses = []
      k_plt = []
      while percent_cost > eps:
            initial = losslaz(x)
            #print(lossq_grad(x))
            x = x - alpha*losslaz_grad(x)
            final = losslaz(x)
            percent_cost = ((abs(initial-final))/initial)*100.0
            print(percent_cost)
            k += 1
            k_plt.append(k)
            losses.append(losslaz(x))
            sq_losses.append(loss(x))
      return x, losses, sq_losses, k_plt 



a, losses, sq_losses, k_plt = linear_regression(theta)

print(a)

plt.scatter(losses, k_plt)
plt.xlabel("Iteration (k)")
plt.ylabel("mean squared Loss")
plt.savefig('mse.png')



