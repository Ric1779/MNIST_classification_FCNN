from matplotlib.pyplot import axis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

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

def initialise_parameters(n,c):
    w = np.random.random((n, c))
    b = np.random.random(c)
    return w,b

def model_forward(X,w,b):
    return X@w + b

def model_backward(m,y_hat,y_hot,X):
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

def fit(X, y, lr, c, epochs):
    m, n = X.shape
    w,b = initialise_parameters(n,c)
    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating hypothesis/prediction.
        z = model_forward(X,w,b)
        y_hat = softmax(z)
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        
        w_grad,b_grad = model_backward(m,y_hat,y_hot,X)
        
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

Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
classes = len(Auto.origin.unique())
print(classes)

# Extract relevant data features
X = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','mpg']].values
Y = Auto[['origin']].values
Y -= 1
Y = Y.flatten()
meann = np.mean(X,axis=0)
stdd = np.std(X,axis=0)
X = (X-meann)/stdd

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )

# Training
w, b, l = fit(X_train, Y_train, lr=0.008, c=3, epochs=3000)

# Accuracy for training set.
train_preds = predict(X_train, w, b)
print(accuracy(Y_train, train_preds))
# got an accuracy of 70%

# Accuracy for test set.
test_preds = predict(X_test, w, b)
print(accuracy(Y_test, test_preds))
# got an accuracy of 65%

print('real value::{}'.format(Y_test[:10]))
print('predicted value:{}'.format(test_preds[:10]))