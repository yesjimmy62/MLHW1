#sigmoid function
from math import exp
import numpy as np

def sigmoid_function(x):
    return 1.0/(1.0+exp(-x))

def sigmoid(z):
    if isinstance(z, list):
        #return list
        sig = range(len(z))
        for i in range(len(z)):
            sig[i] = sigmoid_function(z[i])
        return sig
    elif isinstance(z, np.ndarray):
        #sig = 1.0/(1.0+exp(-z))
        
        if len(z.shape) is 1: # vector
            sig = np.zeros(z.size)
            for i in range(z.size):
                sig[i] = sigmoid_function(z[i])
                
        elif len(z.shape) is 2: # matrix
            sig = np.zeros(z.shape)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    sig[i][j] = sigmoid_function(z[i][j])
        return sig
    elif isinstance(z, float) or isinstance(z, int):
        return sigmoid_function(z)
"""
D_sigmoid(z)
input:
    z: a numpy.array
"""
def D_sigmoid(z):
    if isinstance(z, np.ndarray):
        gradient = np.zeros(z.size)
        for i in range(z.size):
            gradient[i] = sigmoid_function(z[i]) * (1 - sigmoid_function(z[i]));
        return gradient