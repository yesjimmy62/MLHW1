#gradient descent
#Author: yesjimmy62

import CostFunction_SpecialY
reload(CostFunction_SpecialY)
from CostFunction_SpecialY import *
import numpy as np
import random
import Accuracy_SpecialY
reload(Accuracy_SpecialY)
from Accuracy_SpecialY import *
import Output_Weight
reload(Output_Weight)

"""
this is for special form of Y:
    in GradientDescent_Bunch.py
    Y is one-dimensional array, like [1,3,5,4,2...], each elements means the label of the data
    but the special form of this will be:
        [[0, 1, 0, 0, 0, ...]
          0, 0, 0, 1, 0, ...]
          0, 0, 0, 0, 0, 1, ...]
          ...
          ]
    Actually, it may save some time in CostFunction.py
    because in original CostFuction.py, we need to transform Y into this form 
"""
def GradientDescent_Bunch_SpecialY(X, Y, weight, layer_size, learning_rate,iteration, Bunch_size):
    [row_X, col_X] = X.shape
    #X_Bunch = np.zeros([Bunch_size, col_X])
    
    #for output file name
    outfile = 'weight_'
    
    for ite in range(iteration):
        print 'Iteration '+str(ite)+' GradientDescent...Start...\n'
        bunch_sample = random.sample(range(row_X), Bunch_size)
        #bunch_sample = [0]
        X_Bunch = X[bunch_sample]
        Y_Bunch = Y[bunch_sample]
        Cost, weight_gradient = CostFunction_SpecialY(weight, layer_size, X_Bunch, Y_Bunch, 1)

        print '   SampleCost:'+str(Cost)
        print 'Iteration '+str(ite)+' GradientDescent...End...\n'
        print 'Iteration '+str(ite)+' GradientDescent...Update weights...Start...\n'
        for i in range(len(weight)):
            weight[i] -= learning_rate*weight_gradient[i];
        print 'Iteration '+str(ite)+' GradientDescent...Update weights...End...\n'
        #print 'gradient:'+str(weight_gradient[0])
        #[TotalCost, TotalAccuracy] = Accuracy_SpecialY(weight, layer_size, X, Y, 1)
        #print '   TotalCost:'+str(TotalCost)
        #print '   Accuracy: '+str(TotalAccuracy)
        if ite%29 is 0:
            Output_Weight.Output_Weight(weight, layer_size, outfile+str(ite))
        
        
    [TotalCost, TotalAccuracy] = Accuracy_SpecialY(weight, layer_size, X, Y, 1)
    print '   TotalCost:'+str(TotalCost)
    print '   Accuracy: '+str(TotalAccuracy)
    return weight