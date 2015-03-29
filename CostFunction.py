#Cost Function
#Using Cross-Entropy Method

import numpy as np
import sigmoid
reload(sigmoid)
from sigmoid import *

# In the CostFunction, (1) Forward Propagation  (2) BackPropagation
# Need to know the shape of X ---> know the batch size

def CostFunction(weight, layer_size, X, y, Lambda):

    # Write checking for weight and layer_size
    if not isinstance(layer_size, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return
    # Initializing (1)the Cost (2) Weight_Gradient (3) Number of layer (4) batch size
    #              (5) a: input of layers (6) z: output of layers

    Cost = 0.     #(1)
    weight_gradient = range(len(weight))
    for i in range(len(weight)):
        weight_gradient[i] = np.zeros(weight[i].shape)  #(2)

    num_layer = layer_size.size      #(3)
    num_bunch = X.shape[0]           #(4)

    #Calculate Cost: forward propagation

    a = range(num_layer)  # (5)
    z = range(num_layer)  # (6)

    # Data input layer: a[0]

    a[0] = np.concatenate((np.ones([num_bunch ,1]), X), axis=1)
    z[1] = a[0].dot( weight[0].T)
    a[1] = sigmoid(z[1])

    a[1] = np.concatenate((np.ones([num_bunch, 1]),a[1]), axis=1)
    z[2] = a[1].dot(weight[1].T)
    a[2] = sigmoid(z[2])

    y_real = np.zeros([num_bunch, layer_size[-1]])

    for data_index in range(num_bunch):
        label = int(y[data_index]+0.0001)
        if label is 10:
            label = 0
        y_real[data_index][label] = 1
        #to do...
        Cost = Cost - np.log(a[2][data_index]).dot(y_real[data_index])-np.log(1.-a[2][data_index]).dot(1.-y_real[data_index])

    Cost = Cost/num_bunch

    #to do:regularization

    #back-propagation
    #to do: all in matrix form...?
    delta=range(num_layer)
    delta_vec = range(num_layer)
    delta[num_layer-1] = a[num_layer-1] - y_real

    for train_i in range(num_bunch):
        delta_vec[2] = delta[2][train_i].reshape(delta[2][train_i].size,1)
        delta_vec[1] = ((weight[1].T).dot(delta_vec[2]))*((D_sigmoid(np.concatenate(([1],z[1][train_i]), axis=1))).reshape(1+layer_size[1], 1))
        delta_vec[1] = np.array([delta_vec[1][k] for k in range(1, delta_vec[1].size)])

        weight_gradient[1] += delta_vec[2].dot(a[1][train_i].reshape(1,layer_size[1]+1))
        weight_gradient[0] += delta_vec[1].dot(a[0][train_i].reshape(1,layer_size[0]+1))


    weight_gradient[0] /= num_bunch
    weight_gradient[1] /= num_bunch

    #to do:regularization

    #return Cost
    return Cost, weight_gradient
