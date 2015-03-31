#Neural Network


import numpy as np
import math
import time
import sys
import load_everything
import ParameterInitialization
import CostFunction
import GradientDescent
import GradientDescent_Bunch_SpecialY
import GradientDescent_Momentum
import Debug_GradientDescent
reload(load_everything)
reload(ParameterInitialization)
reload(CostFunction)
reload(GradientDescent)
reload(GradientDescent_Bunch_SpecialY)
reload(GradientDescent_Momentum)
reload(Debug_GradientDescent)
from ParameterInitialization import *
from CostFunction import *
from GradientDescent import *
from GradientDescent_Bunch_SpecialY import *
from GradientDescent_Momentum import *
from load_everything import *
from Debug_GradientDescent import *
#reload (load_HandWriting)
#reload(load_HandWriting.load_HandWriting())

#baseline
input_layer_size = 69
hidden_layer_size = 125
output_layer_size = 48

#loading training data

print 'loading training data: Start...'

time0 = time.time()
[dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_nums, train_ids] = load_everything(200)
time1 = time.time()
print 'loading training data: End...spend '+str(time1-time0)+ 'sec.'
sys.stdout.flush()
#weight initialization
weight = range(2)
weight[0] = ParameterInitialization(input_layer_size+1, hidden_layer_size)
weight[1] = ParameterInitialization(hidden_layer_size+1, output_layer_size)

layer_size = np.array([input_layer_size, hidden_layer_size, output_layer_size])

learning_rate = 0.001
iteration = 10
Bunch_size = 128
#final_weight = GradientDescent(X, Y, weight, layer_size, learning_rate,iteration)
final_weight = GradientDescent_Bunch_SpecialY(train_features, train_nums, weight, layer_size, learning_rate, iteration, Bunch_size)

#Debug_GradientDescent_SpecialY(train_features, train_nums, weight, layer_size)

