#Cost Function
#log method
"""
input:
    weight:
        #a 3-dim array
        weight[level][row][column]
    layer_size: 
        layer_size[0]: input_layer_size
        layer_size[-1]:output_layer_size
    y:
        1-dimensional vector
output:
    Cost:
        the cost calculated by specific cost function
    gradient:
        gradient with respect to weight and bias
"""
import numpy as np
import sigmoid
reload(sigmoid)
from sigmoid import *
    
def CostFunction_SpecialY(weight, layer_size, X, Y, Lambda):
    #to do: write checking for weight and layer_size
    """
    if not isinstance(weight, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return 
    """
    if not isinstance(layer_size, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return 
        
    Cost = 0.
    #to do... what is the weight's type???
    #weight_gradient = np.zeros(weight.shape)
    weight_gradient = range(len(weight))
    for i in range(len(weight)):
        weight_gradient[i] = np.zeros(weight[i].shape)
    
    
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
    
    #print 'a[0]:'
    #print a[0][0].reshape(1,layer_size[0]+1)
    """
    print 'a[0][0]'
    print a[0][0]
    print 'a[1][0]'
    print a[1][0]
    print 'a[2][0]'
    print a[2][0]
    """
    for data_index in range(num_bunch):
        #to do...
        #print 'np.log:'+str(a[2][data_index])
        #print ('Y:')+str(Y[data_index])
        Cost = Cost - np.log(a[2][data_index]).dot(Y[data_index])-np.log(1.-a[2][data_index]).dot(1.-Y[data_index])
    
    Cost = Cost/num_bunch
    
    #to do:regularization
    
    #back-propagation
    #to do: all in matrix form...?
    delta=range(num_layer)
    delta_vec = range(num_layer)
    delta[num_layer-1] = a[num_layer-1] - Y
    
    for train_i in range(num_bunch):
        delta_vec[2] = delta[2][train_i].reshape(delta[2][train_i].size,1)
        delta_vec[1] = ((weight[1].T).dot(delta_vec[2]))*((D_sigmoid(np.concatenate(([1],z[1][train_i]), axis=1))).reshape(1+layer_size[1], 1))
        delta_vec[1] = np.array([delta_vec[1][k] for k in range(1, delta_vec[1].size)])
        
        weight_gradient[1] += delta_vec[2].dot(a[1][train_i].reshape(1,layer_size[1]+1))
        weight_gradient[0] += delta_vec[1].dot(a[0][train_i].reshape(1,layer_size[0]+1))
        #print 'weight1:'+str(weight_gradient[1])
        #print 'weight0:'+str(weight_gradient[0])
        
        #print 'weight[0][0]:'
        #print weight_gradient[0][0]
        #print 'delta_vec[1][0]:'
        #print delta_vec[1][0]
        #print 'a[0]:'
        #print a[0][train_i].reshape(1,layer_size[0]+1)
    """
    print 'a[2][0]'
    print a[2][0]
    print 'Y'
    print Y[0]
    print 'delta2'
    print delta_vec[2][0]
    print 'delta1'
    print delta_vec[1][0]
    print 'weight_gradient[0][0]'
    print weight_gradient[0][0]
    weight_gradient[0] /= num_bunch
    weight_gradient[1] /= num_bunch
    """
    #to do:regularization
    
    #return Cost
    return Cost, weight_gradient
        
        
        