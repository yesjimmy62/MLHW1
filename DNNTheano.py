""" DNNTheano.py
The DNN network implemting theano
"""
# standard library
import cPickle # turn python object into serial bytes in C

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample #?????

# Activation functions for neurons
from theano import function
from theano.tensor import tanh

### constants
GPU = True # False to use CPU
if GPU
    print "Run under GPU!"
    try: theano.config.device = 'gpu'
    except: pass # already set
    theano.config.floatX = 'float32'
else:
    print "Run under CPU"

### Load Data

### Main class to construct DNN
class DNNTheano():
    def __init__(self ,layers, mini_batch_size):
        """ takes a 'list' of layers, ex [38,129,49] and value of
        batch size """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x self.mini_batch_size)








### Miscellanea
z_sigmoid_matrix = T.dmatrix('z')
sigmoid_calculation = (1 + T.tanh(z_sigmoid_matrix/2)) / 2
sigmoid_result = function([z_sigmoid_matrix], sigmoid_calculation)

def sigmoid(z):
    return sigmoid_result([[z]])[0][0]

def sigmoid_prime(z):
    """derivative of sigmoid"""
    return sigmoid(z)*(1-sigmoid(z))

""" still not sure about the vector version
sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)
"""
