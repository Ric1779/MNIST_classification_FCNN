import numpy as np
import matplotlib.pyplot as plt
import math
from load_mnist import load_mnist

# Initialize parameters using random initialization
def params_init(layers):
    params = {}
    L = len(layers) 
    for l in range(L-1):
        params['W'+str(l+1)] = (np.random.randn(layers[l],layers[l+1]))                        
        params['b'+str(l+1)] = np.zeros((layers[l+1],1))
        
        assert(params['W' + str(l+1)].shape == (layers[l], layers[l+1]))
        assert(params['b' + str(l+1)].shape == (layers[l+1], 1))
        
    return params
# checking if parameters initialization work
params = params_init([2,3,3])
print("W1 =\n %s" %(params['W1']))
print("b1 =\n %s" %(params['b1']))
print("W2 =\n %s" %(params['W2']))
print("b2 =\n %s" %(params['b2']))