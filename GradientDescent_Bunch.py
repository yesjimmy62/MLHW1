#gradient descent
#Author: yesjimmy62

import CostFunction
reload(CostFunction)
from CostFunction import *
import numpy as np
import random
import Accuracy
reload(Accuracy)
from Accuracy import *
def GradientDescent_Bunch(X, Y, weight, layer_size, learning_rate,iteration, Bunch_size):
    [row_X, col_X] = X.shape
    #X_Bunch = np.zeros([Bunch_size, col_X])
    
    for ite in range(iteration):
        print 'Iteration '+str(ite)+' GradientDescent...Start...\n'
        bunch_sample = random.sample(range(row_X), Bunch_size)
        #bunch_sample = [0]
        X_Bunch = X[bunch_sample]
        Y_Bunch = Y[bunch_sample]

        Cost, weight_gradient = CostFunction(weight, layer_size, X_Bunch, Y_Bunch, 1)
                
        print '   SampleCost:'+str(Cost)
        print 'Iteration '+str(ite)+' GradientDescent...End...\n'
        print 'Iteration '+str(ite)+' GradientDescent...Update weights...Start...\n'
        for i in range(len(weight)):
            weight[i] -= learning_rate*weight_gradient[i];
        print 'Iteration '+str(ite)+' GradientDescent...Update weights...End...\n'
        #[TotalCost, TotalAccuracy] = Accuracy(weight, layer_size, X, Y, 1)
        #print '   TotalCost:'+str(TotalCost)
        #print '   Accuracy: '+str(TotalAccuracy)
        
    [TotalCost, TotalAccuracy] = Accuracy(weight, layer_size, X, Y, 1)
    print '   TotalCost:'+str(TotalCost)
    print '   Accuracy: '+str(TotalAccuracy)
    return weight