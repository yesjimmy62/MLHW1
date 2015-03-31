"""network3.py
~~~~~~~~~~~~~~
A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

In particular, the API is similar to network2.py.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably, 
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
"""

#### Libraries
from theano_load_everything import load_everything as load # our load function

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import softmax

# Activation functions for neurons
def linear(z): return z                   # ???
def ReLU(z): return T.maximum(0.0, z)     # ???
# from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU. Modify theano_DNN_class to change the GPU flag"
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU. Modify theano_DNN_class to change the GPU flag"

#### Load the data
def load_data_shared(train_data_number = 50000,valid_data_number = 10000,test_data_number = 10000):
    train_features, train_features_ans, validation_features, validation_features_ans, test_features, test_features_ans = load(train_data_number,valid_data_number,test_data_number) # change the number to get different train, test data number and validation data number

    training_data = []
    validation_data = []
    test_data = []
    training_data.append(train_features)
    training_data.append(train_features_ans)
    validation_data.append(validation_features)
    validation_data.append(validation_features_ans)
    test_data.append(test_features)
    test_data.append(test_features_ans)

    def shared(data):
        """Place the data into shared variables allowing Theano to copy
        the data to the GPU"""
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network():
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        # take in all parameters from layers to class
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")  
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data # x is input, y is output(ans)
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        ##### DEBUG #####
        # print(size(training_data))
        # print(size(validation_data))
        # print(size(test_data))
        
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) 
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index (int64)
        train_mb = theano.function(
            [i], cost, updates = updates,
            givens={
                self.x:
                training_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                self.y: 
                training_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
                # input and output; ex. 0 - 2000, 2000 - 4000
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y), # to the ast layer
            givens={
                self.x: 
                validation_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                self.y: 
                validation_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
            })
        """test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: 
                test_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size],
                self.y: 
                test_y[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
            })"""
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x: 
                # test_x[i * self.mini_batch_size: (i+1) * self.mini_batch_size]
                test_x[i * 0: num_test_batches * self.mini_batch_size] # ALL , nick
                # test_x # all test data...... Nick
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                # num_training_batches => how many batches for total training data
                iteration = num_training_batches * epoch + minibatch_index
                # iteration = how many mini-batches
                if iteration % 1000 == 0: 
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0: # one epoch passed
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}:\nvalidation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        #if test_data:
                            #test_prediction = test_mb_predictions(0) # NICK
                        """test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))"""
        #print(test_prediction) #NICK
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        # print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types
class SoftmaxLayer():
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype = theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype = theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b] # parameters are weights and biases

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.y_out_vector = self.output # NICK
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class FullyConnectedLayer():
    def __init__(self, n_in, n_out,activation_fn, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
