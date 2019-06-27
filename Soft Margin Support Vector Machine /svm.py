import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

import time
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

C = 10.0
eps = 0.04
rho = 0.000000001

display_step = 500
#load training data
def load_data(path):
      features = []
      labels = []
      metadata = []
      file_to_parse = open(path, "r")
      lines = file_to_parse.readlines()  
      iter = 0
      for x in lines:
            data = x.split()
            if data[0][0] == '@':
                  metadata.append(x)
            else:
                  features.append(np.array(data[0:8]))
                  labels.append(np.array(data[8]))
            iter+=1
      featuresf = []
      labelsf = []
      for i in range(len(features)):
            labelsf.append(int(labels[i]))
            featuresf.append([])
            for j in range(len(features[i])):
                  featuresf[i].append(float(features[i][j]))
      return np.array(featuresf), np.array(labelsf), metadata
features, labels, metadata = load_data('data.txt')
num_data = len(features)
dim = len(features[0])

#parameters
w = np.zeros(dim)
b = 0.0

def shuffle(features, lables):
      x = np.arange(0, len(features))
      np.random.shuffle(x)
      new_features = []
      new_lables = []
      for i in range(len(x)):
            new_features.append(features[x[i]])
            new_lables.append(labels[x[i]])
      return new_features, new_lables

def make_batches(x, y, batch_size):
      batch_x = []
      batch_y = []
      for i in range(0, len(x) , batch_size):
            batch_x.append(x[i:i+batch_size])
            batch_y.append(y[i:i+batch_size])
      return batch_x, batch_y


#relu activation to classify 
def relu(x):
      return max(0, x)

#soft margin support vector machine
def soft_margin_svm(x, y, w, b):
      v1 = 1.0 - y*(np.matmul(np.transpose(w), x) + b)
      return relu(v1)

#cost function to be optimized 
def cost(x, y, w, b):
      var = 0.0
      for i in range(len(x)):
            var += soft_margin_svm(x[i], y[i], w, b)
      return (0.5*np.linalg.norm(w))+(C*var)

def single_cost(x, y, w, b):
      return (0.5*np.linalg.norm(w))+(C*soft_margin_svm(x, y, w, b))

#partial of cost with respect to weights (Batch Grad Desc)
def dCost_dw(x, y, w, b):
      var = np.zeros(w.shape)
      for i in range(len(x)):
            if y[i]*(np.matmul(np.transpose(w), x[i]) + b) < 1:
                  var += (-y[i]*x[i])
      return w + C*var

#partial of cost with respect to bias (Batch Grad Desc)
def dCost_db(x, y, w, b):
      var = np.zeros(w.shape)
      for i in range(len(x)):
            if y[i]*(np.matmul(np.transpose(w), x[i]) + b) < 1:
                  var += 1.0
      return (w + C*var)[0]

#partial of cost with respect to weights (Stochastic GD)
def dcost_dw(x, y, w, b):
      if y*(np.matmul(np.transpose(w), x) + b) < 1:
            return (w + C*(-y*x))
      else: 
            return w

#partial of cost with respect to bias (Stochastic GD)
def dcost_db(x, y, w, b):
      if y*(np.matmul(np.transpose(w), x) + b) < 1:
            return (w + C)[0]
      else: 
            return w[0]

#convergence criteria 
def percent_del_cost(cost_initial, cost_final):
      return (abs(cost_initial - cost_final)/cost_initial)*100.0

#train support vector machine parameters
def batch_gradient_descent(x, y, w, b):
      eps = 0.04
      eta =  0.000000001
      k = 0
      k_plt = []
      losses = []
      BGD_START_time = time.time()
      while(True):
            cost_initial = cost(x, y, w, b)
            k_plt.append(k)
            losses.append(cost_initial)
            w -= (eta * dCost_dw(x, y, w, b))
            b -= (eta * dCost_db(x, y, w, b))            
            cost_final = cost(x, y, w, b)
            k+=1
            pdc = percent_del_cost(cost_initial, cost_final)
            if k%display_step == 0:
                  print(pdc)
            if pdc < eps:
                  print("BGD SUCCESS!!")
                  break
      print("MBD --- %s seconds ---" % (time.time() - BGD_START_time))
      return w, b, k_plt, losses


#train one example at a time
def stochastic_gradient_descent(x, y, w, b):
      eps = 0.0003
      eta = 0.000000004
      k = 0
      k_plt = []
      losses = []
      del_initial = 0.0
      SGD_START_time = time.time()
      while(True):
            k_plt.append(k)
            loss_initial = cost(x, y, w, b)
            losses.append(loss_initial)
            #update weights one data at a time 
            for j in range(num_data):
                  w -= (eta * dcost_dw(x[j], y[j], w, b))
                  b -= (eta * dcost_db(x[j], y[j], w, b))       
            loss_final = cost(x, y, w, b)
            percent_del = percent_del_cost(loss_initial, loss_final)
            del_final = 0.5*del_initial + 0.5*percent_del
            del_initial = del_final
            if k%display_step == 0:
                  print(k, del_final)
            k+=1
            if del_final < eps:
                  print("SGD SUCCESS!!")
                  break
      print("SGD --- %s seconds ---" % (time.time() - SGD_START_time))
      return w, b, k_plt, losses


def minibatch_gradient_descent(x, y, w, b):
      eps = 0.004
      eta =  0.000000001
      batch_size = 40
      k = 0
      k_plt = []
      losses = []
      del_initial = 0.0
      #x, y = shuffle(x, y)
      x_batches, y_batches = make_batches(x, y, batch_size)
      
      MBGD_START_time = time.time()
      while(True):
            k_plt.append(k)
            loss_initial = cost(x, y, w, b)
            losses.append(loss_initial)
            #update weight every batch
            for j in range(len(x_batches)):
                  w -= (eta * dCost_dw(x_batches[j], y_batches[j], w, b))
                  b -= (eta * dCost_db(x_batches[j], y_batches[j], w, b))       
            loss_final = cost(x, y, w, b)
            percent_del = percent_del_cost(loss_initial, loss_final)
            del_final = 0.5*del_initial + 0.5*percent_del
            del_initial = del_final
            if k%display_step == 0:
                  print(k, del_final)
            k+=1
            if del_final < eps:
                  print("MBGD SUCCESS!!")
                  break
      print("MBGD --- %s seconds ---" % (time.time() - MBGD_START_time))
      return w, b, k_plt, losses

w = np.zeros(dim)
b = 0.0
nw, nb, k_plt0, losses0 = batch_gradient_descent(features, labels, w, b)

w1 = np.zeros(dim)
b1 = 0.0
nw1, nb1, k_plt1, losses1 = stochastic_gradient_descent(features, labels, w1, b1)

w2 = np.zeros(dim)
b2 = 0.0
nw2, nb2, k_plt2, losses2 = minibatch_gradient_descent(features, labels, w2, b2)

print("SIZES")
print(k_plt0)
print(k_plt1)
print(k_plt2)


ax.plot(k_plt0, losses0, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
ax.plot(k_plt1, losses1, marker='', color='olive', linewidth=2)
ax.plot(k_plt2, losses2, marker='', color='olive', linewidth=2, linestyle='dashed')

plt.legend(['Batch GD','Stochastic GD', 'MiniBatch GD'], loc='upper right')

plt.xlabel("Iteration (k)")
plt.ylabel("Loss")
plt.title("Losses")
plt.savefig("digarams.png")


