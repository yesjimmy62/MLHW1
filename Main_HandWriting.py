#Neural Network


import numpy as np
import math
import load_HandWriting
import display_HandWriting
import ParameterInitialization
import CostFunction
import GradientDescent
import GradientDescent_Bunch
import GradientDescent_Momentum
import HandWritingPrediction
reload(load_HandWriting)
reload(display_HandWriting)
reload(ParameterInitialization)
reload(CostFunction)
reload(GradientDescent)
reload(GradientDescent_Bunch)
reload(GradientDescent_Momentum)
reload(HandWritingPrediction)
from load_HandWriting import *
from display_HandWriting import *
from ParameterInitialization import *
from CostFunction import *
from GradientDescent import *
from GradientDescent_Bunch import *
from GradientDescent_Momentum import *
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

layer_size = np.array([input_layer_size, hidden_layer_size, output_layer_size])
#CostFunction(weight, layer_size, X, Y, 1)

learning_rate = 2
iteration = 150
Bunch_size = 15
#final_weight = GradientDescent(X, Y, weight, layer_size, learning_rate,iteration)
#final_weight = GradientDescent_Bunch(X, Y, weight, layer_size, learning_rate, iteration, Bunch_size)

learning_rate = 0.9
momentum = 0.3
iteration = 150
Bunch_size = 15
decay_rate = 0.9999
final_weight = GradientDescent_Momentum(X, Y, weight, layer_size, learning_rate, decay_rate, momentum,iteration, Bunch_size)


HandWritingPrediction(X, Y, final_weight, layer_size)