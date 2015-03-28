#Momentum Gradient Descent
#Author: yesjimmy62

#gradient descent
import CostFunction 
reload(CostFunction)
from CostFunction import *
import numpy as np
import random
import Accuracy
reload(Accuracy)
from Accuracy import *
def GradientDescent_Momentum(X, Y, weight, layer_size, learning_rate, decay_rate, momentum_coeff, iteration, Bunch_size):
    [row_X, col_X] = X.shape
    #X_Bunch = np.zeros([Bunch_size, col_X])
    
    weight_gradient = range(2) #to update alternately
    #initial the [0] first, [1] will be assigned in for loop
    weight_gradient[0] = range(len(weight))
    for i in range(len(weight)):
        weight_gradient[0][i] = np.zeros(weight[i].shape)
        
    for ite in range(iteration):
        print 'Iteration '+str(ite)+' GradientDescent...\n'
        bunch_sample = random.sample(range(row_X), Bunch_size)
        X_Bunch = X[bunch_sample]
        Y_Bunch = Y[bunch_sample]
        Cost, weight_gradient[(ite+1)%2] = CostFunction(weight, layer_size, X_Bunch, Y_Bunch, 1)
        #print '   SampleCost:'+str(Cost)
        
        #update the weight: momentum method
        for i in range(len(weight)):
            weight[i] += momentum_coeff*weight_gradient[ite%2][i]-learning_rate * weight_gradient[(ite+1)%2][i];
            
        #learning rate decay:
        learning_rate *= decay_rate
        
        #[TotalCost, TotalAccuracy] = Accuracy(weight, layer_size, X, Y, 1)
        #print '   TotalCost:'+str(TotalCost)
        #print '   Accuracy: '+str(TotalAccuracy)
        
    [TotalCost, TotalAccuracy] = Accuracy(weight, layer_size, X, Y, 1)
    print '   TotalCost:'+str(TotalCost)
    print '   Accuracy: '+str(TotalAccuracy)
    return weight