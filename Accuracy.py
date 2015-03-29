#Accuracy
#log method
import numpy as np
import sigmoid
reload(sigmoid)
from sigmoid import *

def Accuracy(weight, layer_size, X, y, Lambda):
    #to do: write checking for weight and layer_size
    if not isinstance(layer_size, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return

    Cost = 0.
    #to do... what is the weight's type???

    num_layer = layer_size.size
    num_bunch = X.shape[0]   #number of data in a bunch
    #Calculate Cost: forward propagation
    #to do: bunch??, change to adaptive layer number

    a = range(num_layer)
    z = range(num_layer)

    #input layer: a[0]
    a[0] = np.concatenate((np.ones([num_bunch ,1]), X), axis=1)
    z[1] = a[0].dot( weight[0].T)
    a[1] = sigmoid(z[1])

    a[1] = np.concatenate((np.ones([num_bunch, 1]),a[1]), axis=1)
    z[2] = a[1].dot(weight[1].T)
    a[2] = sigmoid(z[2])

    y_real = np.zeros([num_bunch, layer_size[-1]])
    num_correct = 0
    for data_index in range(num_bunch):
        label = int(y[data_index]+0.0001)
        if label is 10:
            label = 0
        y_real[data_index][label] = 1
        #to do...
        Cost = Cost - np.log(a[2][data_index]).dot(y_real[data_index])-np.log(1-a[2][data_index]).dot(1-y_real[data_index])

        if int(a[2][data_index].argmax()) is label:
            num_correct += 1

    print 'num_correct:'+str(num_correct)+', num_bunch:'+str(num_bunch)
    Accuracy = float(num_correct)/float(num_bunch)
    Cost = Cost/num_bunch
    return [Cost, Accuracy]

