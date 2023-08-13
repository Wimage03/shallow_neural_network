## NEURAL NETWORK IMPLEMENTATION (1 hidden layer)
import numpy as np
import copy
# X, Y are datasets 

# reshape data to desired dimensions
# standardize data

# layer selection
def layer_set(X, Y):
  n_x = X.shape[0]
  n_h = 4
  n_y = Y.shape[0]

  return n_x, n_h, n_y

# initialize parameters
n_x, n_h, n_y = layer_set(X,Y)
def initialize_parameters(n_x, n_h, n_y):
  W1 = np.random.randn(n_h, n_x)
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h)
  b2 = np.zeros((n_y, 1))

  parameters = {'W1':W1,
                'W2':W2,
                'b1':b1,
                'b2':b2}

  return parameters
  
# forward-propagation
def sigmoid(z):
  s = 1/(1+np.exp(-z))
  return s

def forward_propagation(X, parameters):
  W1 = parameters['W1']
  W2 = parameters['W2']
  b1 = parameters['b1']
  b2 = parameters['b2']
  
  Z1 = np.dot(W1, X) + b1
  A1 = np.tanh(Z1)
  Z2 = np.dot(W2, A1) + b2
  A2 = sigmoid(Z2)

  cache = {'Z1':Z1,
           'Z2':Z2,
           'A1':A1,
           'A2':A2}
  return A2, cache

# computing cost

def computing_cost(Y, cache):
  A2 = cache['A2']
  m = Y.shape[1]
  logprobs = np.multiply(np.log(A2), Y) + np.multiply(1-Y, np.log(1-A2))
  cost = (-1/m) * np.sum(logprobs)

  return cost

# backward-propagation
def backward_propagation(X, Y, parameters, cache):
  m = X.shape[1]
  
  W1 = copy.deepcopy(parameters['W1'])
  W2 = copy.deepcopy(parameters['W2'])
  A1, A2 = cache['A1'], cache['A2']
  dZ2 = A2 - Y
  dW2 = (1/m) * np.dot(dZ2, A1.T)
  db2 = (1/m) * np.sum(dZ2)

  dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
  dW1 = (1/m) * np.dot(dZ1, X.T)
  db1 = np.sum(dZ1)

  grads = {'dW2':dW2,
           'dW1':dW1,
           'db1':db1,
           'db2':db2}

  return grads

# gradient descent
def update_parameters(parameters, grads, learning_rate = 1.2):

  W1, W2, b1, b2 = parameters['W1'], parameters['W1'], parameters['b1'], parameters['b2']

  dW1, dW2, db1, db2 = grads['dW1'], grads['dW2'], grads['db1'], grads['db2']
  
  W1 = W1 - (learning_rate * dW1)
  W2 = W2 - (learning_rate * dW2)
  b1 = b1 - (learning_rate * db1)
  b2 = b2 - (learning_rate * db2)

  parameters = {'W1':W1,
                'W2':W2,
                'b1':b1,
                'b2':b2}
  return parameters

# put everything together!!

def nn_model(X, Y, num_iterations = 10000, print_cost = False):
  n_x, n_h, n_y = layer_set(X, Y)
  parameters = initialize_parameters(n_x, n_h, n_y)

  for i in range(num_iterations):
    A2, cache = forward_propagation(X, parameters)
    cost = computing_cost(Y, cache)
    grads = backward_propagation(X, Y, parameters, cache)
    parameters = update_parameters(parameters, grads, learning_rate = 1.2) 

  return parameters

# NOW WE PREDICT
def predict(X, parameters):
  m = X.shape[1]
  predictions = np.zeros((1,m))
  A2, cache = forward_propagation(X, parameters)

  for i in range(A2.shape[1]):
    if A2[0, i] > 0.5:
      predictions[0, i] = 1
    else:
      predictions[0, i] = 0
  return predictions