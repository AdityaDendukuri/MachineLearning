'''
Basic Numpy 3 Layer Neural Network by Aditya Dendukuri 
'''

import numpy as np
import matplotlib.pyplot as plt 
import csv
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)


class NEURAL_NETWORK:
      def __init__(self, structure):
            self.weights = []
            self.features = []
            self.lables = []
            self.outputs = []
            self.lmbd = 3
            for i in range(len(structure)-1):
                  self.weights.append(np.random.randn(structure[i+1], structure[i]))   
            self.weights[0] = self.load_weights('data/initial_W1.csv')
            self.weights[1] = self.load_weights('data/initial_W2.csv')
            self.num_layers = len(self.weights)

      def __call__(self, x):
            return self.predict(self.model(x))

      def model(self, x, bias=True):
            output = np.copy(x)
            self.outputs = []
            for i in range(len(self.weights)):
                  #append a 1 to the input for bias
                  if bias:
                        output = self.add_bias_node(output)
                  output = self.propogate_layer(i, output)
                  self.outputs.append(output)
            return output

      def loss(self, y_true, y_pred):
            return - y_true*np.log(y_pred) - (1.-y_true)*np.log(1.-y_pred)
      
      def predict(self, x):
            output = self.model(x).flatten()
            y = np.max(output)
            index = int(list(output).index(y))
            return index+1

      def propogate_layer(self, i, x):
            v1 = np.matmul(x, np.transpose(self.weights[i]))
            return np.array([self.sigmoid(v1)])

      def batch_loss(self, y, f):
            var1 = 0.0
            for i in range(len(f)):
                  
                  y_pred = self.model(f[i])
                  for j in range(len(y[i])):
                        var1 += self.loss(y[i][j], y_pred[0][j])
            var2 = 0.0
            for W in self.weights:
                  var2 += np.sum(np.square(W))
            return (1./self.num_features)*var1 + (self.lmbd/(2.*self.num_features))*var2          


      def add_bias_node(self, x):
            return np.insert(np.copy(x), 0, 1.)

      def train(self, train_features, train_lables, learning_rate=0.2, num_steps=1, display_step = 10):
            m = (1./len(train_features))
            print(m)
            losses = []
            for i in range(num_steps):
                  losses.append(nn.batch_loss(train_lables, train_features))
                  grad1 = np.zeros(self.weights[0].shape)
                  grad2 = np.zeros(self.weights[1].shape)
                  #batch gradient desent 
                  for iter, x in enumerate(train_features):
                        #run the neural network on training example
                        predictions = self.model(x)
                        cost_2 = np.subtract(predictions, train_lables[iter])
                        cost_1 = np.multiply(np.matmul(cost_2, self.weights[1][:,1:]), map(self.sigmoid_grad, map(self.logit, self.outputs[0]))) 
                        #calculate partial gradients
                        H = np.insert(self.outputs[0], 0, 1., axis=1)
                        X = np.insert(x, 0, 1., axis=1)
                        del_2 = np.transpose(cost_2) * H
                        del_1 = np.transpose(cost_1) * X
                        #calculate batch gradients
                        grad1 = np.add(grad1, m*del_1)
                        grad2 = np.add(grad2, m*del_2) 
                        iter += 1
                  #Set the first columns to zero 
                  W1_reduced = np.copy(self.weights[0])
                  W2_reduced = np.copy(self.weights[1])
                  W1_reduced[:,0] = 0.
                  W2_reduced[:,0] = 0.
                  grad1 = np.add(grad1, m* self.lmbd * W1_reduced)
                  grad2 = np.add(grad2, m* self.lmbd * W2_reduced)
                  #print(grad1)
                  #print(grad2)
                  #update weights
                  self.weights[0] = np.subtract(np.copy(self.weights[0]) , learning_rate*grad1)
                  self.weights[1] = np.subtract(np.copy(self.weights[1]) , learning_rate*grad2)
                  if i% display_step == 0:
                        print("Loss at iteration: ", i, losses[i])
            np.savetxt('trained_weights1.txt', self.weights[0])
            np.savetxt('trained_weights2.txt', self.weights[1])
            return losses

          
      def load_weights(self, path):
            weights = []
            with open(path) as csvfile:
                  readCSV = csv.reader(csvfile, delimiter=',')
                  for row in readCSV:
                        weights.append(np.fromiter([float(r) for r in row], float))
            return np.array(weights)

      def calc_accuracy(self, f, l_):
            predictions = []
            correct = []
            predictions = map(self.predict, f)
            for i in range(len(predictions)):
                  print(predictions[i], ", ", l_[i])
                  if predictions[i] == l_[i]:
                        correct.append(1.)
                  else:
                        correct.append(0.)
            return (np.sum(correct)/len(correct))*100.

      def sigmoid(self, x):
            return 1./(1. + np.exp(-x))

      def logit(self, x):
            return np.log(x/(1.-x))

      def sigmoid_grad(self, x):
            v = self.sigmoid(x)
            return v*(1-v)

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
            l = []
            for i in range(len(labels)):
                  x = np.zeros(10)
                  x[int(labels[i]) - 1] = 1.
                  l.append(x)
            self.features = features
            self.lables = l
            return self.features, self.lables, labels



nn = NEURAL_NETWORK([400, 25, 500])
f, l, l_ = nn.read_data('data/X.csv', 'data/Y.csv')



losses = nn.train(f, l, num_steps=500)
print("ACCURACY: %d", nn.calc_accuracy(f, l_))
ax.scatter(np.arange(0, len(losses), 1), losses)
plt.savefig('loss_iteration.png')
