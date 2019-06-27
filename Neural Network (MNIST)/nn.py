'''
Basic Numpy Neural Network by Aditya Dendukuri 
'''

import numpy as np
import matplotlib.pyplot as plt 

#logistic function
def sigmoid(x):
      return 1./(1. + np.exp(-x))

def sigmoid_grad(x):
      sig = sigmoid(x)
      return sig * (1. - sig)

class LAYER:
      def __init__(self, id, nn, num_input, num_output):
            self.W =np.random.randn(num_output, num_input+1)
            self.id = id
            self.next = 0
            self.num_data = 0
            self.next_layer = None
            self.num_training = 0
            self.num_testing = 0 
            self.nn = nn
      def __call__(self, x):
            return self.model(x, True)
            
      def model(self, x, bias):
            #append a 1 to the input for bias
            if bias:
                  var = np.append(np.copy(x), [1.])
            else:
                  var = np.append(np.copy(x), [0.])
            v1 = np.matmul(var, np.transpose(self.W))
            
            return np.array([map(sigmoid, v1)])

      def attach(self, layer):
            self.next_layer = layer 
      
      def cost(self, x, predictions, y_true):
            if self.next_layer == None:
                  return np.subtract(y_true , predictions)
            else:
                  n_cost = self.next_layer.cost(self.model(x, True), predictions, y_true)
                  return np.matmul(n_cost, self.next_layer.W[:,:-1])
                  
      def grad(self, x, y_true):
            #if last layer        
            predictions = self.nn(x)    
            
            if self.next_layer == None:
                  var = self.model(x, True)
                  beta = self.cost(var, predictions, y_true)
                  return np.matmul(np.transpose(beta), x)
            else:
                  var = self.model(x, True)
                  sig_grad = np.array(map(sigmoid_grad, var))
                  beta = np.multiply(self.cost(x, predictions, y_true), sig_grad)
                  return np.matmul(np.transpose(beta), x)
                  

class NEURAL_NETWORK:
      def __init__(self, structure):
            self.layers = []
            self.features = []
            self.lables = []
            self.num_features = 0
            self.dim = 0
            self.current_loss = 0.
            self.lmbd = 3
            self.num_classes = 0
            for i in range(len(structure)-1):
                  self.layers.append(LAYER(i, self, structure[i], structure[i+1]))
            for i in range(len(self.layers)-1):
                  self.layers[i].attach(self.layers[i+1])

      def __call__(self, x):
            return self.front_propogate(x)
      
      def front_propogate(self, x):
            var = x
            for i in range(len(self.layers)):
                  var = self.layers[i](var)
            return var     

      def read_data(self, path_X, path_Y):
            features = []
            labels = []
            file_to_parseX = open(path_X, "r")
            file_to_parseY = open(path_Y, "r")
            linesX = file_to_parseX.readlines()
            for x in linesX:
                  data = x.split(',')
                  arr = [float(a) for a in data]
                  features.append(np.reshape(np.array(arr), [1, len(arr)]))
            linesY = file_to_parseY.readlines()
            for x in linesY:
                  data = x.split(',')
                  arr = [float(a) for a in data]
                  labels.append(np.reshape(np.array(arr), [1, len(arr)]))
            self.num_features = len(features)
            self.dim = len(features[0])
            self.features = features
            self.lables = labels
            return np.array(features), np.array(labels)

      def loss(self, x):
            return - x*np.log(x) - (1-x)*np.log(1-x)

      def batch_loss(self, y):
            var1 = np.sum([map(self.loss, x) for x in y])
            var2 = 0.0
            for layer in self.layers:
                  var2 += np.sum(np.square(layer.W))
            return var1 + (self.lmbd/(2.*self.num_features))*var2
      
      def backprop(self, x, y):
            m = 1./self.num_features
            var = np.copy(x)
            var1 = []
            for layer in self.layers:
                  layer_grad = layer.grad(var, y)
                  layer_grad = np.insert(x, len(x[0]), 0., axis=1)
                  var1.append(m*(layer_grad + layer.W))  
            return np.array(var1)

      def train(self, train_features, train_lables, learning_rate = 0.2,iterations=500):
            for _ in range(iterations):
                  grad = np.zeros(shape=[len(self.layers), 1])
                  #calculate gradient
                  for i in range(len(train_features)):
                        grad += self.backprop(train_features[i], train_lables[i])
                  #update weights
                  for i in range(self.layers):
                        self.layers[i].W -= learning_rate*(grad[i])
                        

nn = NEURAL_NETWORK([400, 25, 10])
f, l = nn.read_data('data/X.csv', 'data/Y.csv')
lables = []
for i in range(len(l)):
      x = np.zeros(10)
      x[int(l[i]-1)] = 1.
      lables.append(x)

#print(nn(f[0]))
nn.train(f, lables, iterations=1)
