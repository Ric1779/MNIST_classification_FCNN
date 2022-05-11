from load_mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import math

def one_hot(y, c):
    y_hot = np.zeros((len(y), c))
    y_hot[np.arange(len(y)), y] = 1
    return y_hot

def softmax(z):
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])   
    return exp

def initialise_parameters(model_state,layer_size):
    for i in range(0,len(layer_size)-1):
        model_state['parameters']['layer'+str(i+1)+'_w'] = np.random.random((layer_size[i],layer_size[i+1]))
        model_state['parameters']['layer'+str(i+1)+'_b'] = np.random.random(layer_size[i+1])
    return model_state

def linear_forward(X,w,b):
    return X@w + b

def activation_forward(z,activation):
    q = z
    if activation=="relu":
        q[q<0] = 0
    elif activation=="sigmoid":
        q = 1/(1+(math.e**-z))
    elif activation=="softmax":
        q = softmax(z)
    return q

def linear_backward(layer_number,model_state):
    return model_state['forward_outputs']['layer'+str(layer_number-1)] 

def sigmoid_backward(a):
    ## change
    return a*(1-a)

def relu_backward(a):
    ## change
    q = a
    q[q<0] = 0
    q[q>0] = 1
    return q

def softmax_backward(y_hat,y_hot):
    return y_hat-y_hot

def softmax_backwarddddd(X,y_hat,y_hot):
    m = len(y_hat)
    print('m:{}'.format(m))
    w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
    b_grad = (1/m)*np.sum(y_hat - y_hot)
    return w_grad,b_grad

def activation_backward(layer,a):
    if layer == "softmax":
        act_grad = softmax_backward(y_hat,y_hot)
    elif layer == "sigmoid":
        act_grad = sigmoid_backward(a)
    elif layer == "relu":
        act_grad = relu_backward(a)
    linear_grad = 


def model_forward(X,activation_fn,model_state):
    model_state['forward_outputs']['layer'+str(0)] = X
    for i,act in enumerate(activation_fn):
        model_state['forward_outputs']['layer'+str(i+1)] = \
            linear_forward(\
            model_state['forward_outputs']['layer'+str(i)]\
            ,model_state['parameters']['layer'+str(i+1)+'_w']\
            ,model_state['parameters']['layer'+str(i+1)+'_b'])
        model_state['forward_outputs'][act+str(i+1)] = \
            activation_forward(\
            model_state['forward_outputs']\
            ['layer'+str(i+1)],act)
    return model_state

def model_backward(m,y_hot,model_state,activation_fn):
    l = len(activation_fn)
    for i,layer in enumerate(activation_fn):
        if layer == "softmax":
            model_state['grad'][layer+str(l-i)] = softmax_backward(\
                model_state['forward_outputs'][layer+str(l-i)],y_hot)
            model_state['grad']['layer'+str(l-i)] = 
        elif layer == "relu":
            model_state['grad'][layer+str(l-i)] = relu_backward()


def modell_backward(m,y_hat,y_hot,X):
    # Calculating the gradient of loss w.r.t w and b.
    w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
    b_grad = (1/m)*np.sum(y_hat - y_hot)
    return w_grad,b_grad

def update_parameters(w,b,w_grad,b_grad,alpha):
    # Updating the parameters.
    w = w - alpha*w_grad
    b = b - alpha*b_grad
    return w,b

def compute_cost(y_hat,y):
    return -np.mean(np.log(y_hat[np.arange(len(y)), y]))

def fit(X, y, lr, c,layer_size, activation_fn, epochs):
    model_state = {'forward_outputs':{},
               'grad':{},
               'parameters':{}}
    m, n = X.shape
    model_state = initialise_parameters(model_state,layer_size)
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating prediction.
        model_state = model_forward(X,activation_fn,model_state)
        
        # One-hot encoding y.
        y_hot = one_hot(y, c)

        # Retrieving the softmax output
        y_hat = model_state['forward_outputs']['softmax'+str(len(layer_size)-1)]
        
        model_state = model_backward(m,y_hot,model_state,activation_fn)
        
        w,b = update_parameters(w,b,w_grad,b_grad,lr)
        
        loss = compute_cost(y_hat,y)
        losses.append(loss)
        # Printing out the loss at every 100th iteration.
        if epoch%100==0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))
    return w, b, losses

def predict(X, w, b):
    z = model_forward(X,w,b)
    y_hat = softmax(z)
    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)


#X_train,Y_train,X_test,Y_test = load_mnist()
layer_size = []
activation_fn = []

depth = int(input("Enter the number of layers (including the input layer):"))
print("Enter the size of each layer:")
for i in range(0,depth):
    print("Enter size of layer {}:".format(i+1))
    le = int(input())
    layer_size.append(le)
print('Input layer is marked as layer 0 & the last layer is the softmax layer')
print("Input and the last layer doesn't have an activation function")
print("Enter the activation function for each hidden layer:")
for i in range(1,depth-1):
    print("Enter the activation function after hidden layer {}".format(i))
    le = input()
    activation_fn.append(le)
activation_fn.append('softmax')
print(layer_size)
print(activation_fn)

model_state = {'forward_outputs':{},
               'grad':{},
               'parameters':{}}
X = [[1,1],[2,1],[4,7]]
model_state = initialise_parameters(model_state,layer_size)
model_state = model_forward(X,activation_fn,model_state)
print(model_state)


