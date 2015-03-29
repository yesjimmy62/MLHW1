#gradient descent
import CostFunction
reload(CostFunction)
from CostFunction import *

def GradientDescent(X, Y, weight, layer_size, learning_rate,iteration):
    for ite in range(iteration):
        print 'Iteration ' + str(ite) + ' GradientDescent...\n'
        Cost, weight_gradient = CostFunction(weight, layer_size, X, Y, 1)
        print '   Cost:' + str(Cost)
        for i in range(len(weight)):
            weight[i] -= learning_rate * weight_gradient[i];

    return weight
