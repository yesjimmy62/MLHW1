#Neural Network


import numpy as np
import math
import copy
import load_HandWriting
import display_HandWriting
import ParameterInitialization
import CostFunction
import GradientDescent
import GradientDescent_Bunch
import GradientDescent_Momentum
import GradientDescent_Bunch_SpecialY
import HandWritingPrediction
import Debug_GradientDescent
reload(load_HandWriting)
reload(display_HandWriting)
reload(ParameterInitialization)
reload(CostFunction)
reload(GradientDescent)
reload(GradientDescent_Bunch)
reload(GradientDescent_Momentum)
reload(GradientDescent_Bunch_SpecialY)
reload(Debug_GradientDescent)
reload(HandWritingPrediction)
from load_HandWriting import *
from display_HandWriting import *
from ParameterInitialization import *
from CostFunction import *
from GradientDescent import *
from GradientDescent_Bunch import *
from GradientDescent_Momentum import *
from GradientDescent_Bunch_SpecialY import *
from Debug_GradientDescent import *
from HandWritingPrediction import *
#reload (load_HandWriting)
#reload(load_HandWriting.load_HandWriting())

input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10

[X, Y] = load_HandWriting()

#display_HandWriting(X)

#initialization
#weight=np.array([0,1])
weight = range(2)
weight[0] = ParameterInitialization(input_layer_size+1, hidden_layer_size)
weight[1] = ParameterInitialization(hidden_layer_size+1, output_layer_size)

weight_copy = copy.deepcopy(weight)

layer_size = np.array([input_layer_size, hidden_layer_size, output_layer_size])


learning_rate = 0.1
iteration = 3
Bunch_size = 10
#print 'true weight[0]:'
#print weight[0][0]

#final_weight = GradientDescent(X, Y, weight, layer_size, learning_rate,iteration)
#final_weight = GradientDescent_Bunch(X, Y, weight, layer_size, learning_rate, iteration, Bunch_size)
#Debug_GradientDescent(X, Y, weight, layer_size)

#print 'true weight[0]:'
#print weight[0][0]

"""
learning_rate = 0.9
momentum = 0.3
iteration = 150
Bunch_size = 15
decay_rate = 0.9999
final_weight = GradientDescent_Momentum(X, Y, weight, layer_size, learning_rate, decay_rate, momentum,iteration, Bunch_size)
"""

iteration = 500
Y_real = np.zeros([Y.shape[0], layer_size[-1]])
for data_index in range(Y.shape[0]):
    label = int(Y[data_index]+0.0001)
    if label is 10:
        label = 0
    Y_real[data_index][label] = 1
final_weight = GradientDescent_Bunch_SpecialY(X, Y_real, weight, layer_size, learning_rate, iteration, Bunch_size)

#Debug_GradientDescent_SpecialY(X, Y_real, weight, layer_size)

#HandWritingPrediction(X, Y, final_weight, layer_size)