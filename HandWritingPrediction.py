import numpy as np
from sigmoid import *
import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm
def HandWritingPrediction(X, Y, weight, layer_size):
    num_data = X.shape[0]
    sample_list = range(num_data)
    num_layer = layer_size.size
    a = range(num_layer)
    z = range(num_layer)
    #input layer: a[0]
    num_bunch = 1
    
    while True:
        sampling_data = random.sample(sample_list, 1)
        x = X[sampling_data]
        a[0] = np.concatenate((np.ones([num_bunch ,1]), x), axis=1)
        z[1] = a[0].dot( weight[0].T)
        a[1] = sigmoid(z[1])
        
        a[1] = np.concatenate((np.ones([num_bunch, 1]),a[1]), axis=1)
        z[2] = a[1].dot(weight[1].T)
        a[2] = sigmoid(z[2])
        
        x=x.reshape(20,20)
        x_map = np.zeros([20,20])
        for i in range(20):
            for j in range(20):
                x_map[i][j] = x[19-i][j]
        x_map = zip(*x_map[::-1])
        print 'Prediction:'+str(a[2].argmax())
        plt.imshow(x_map, cmap = cm.Greys_r)
        plt.show()
        
        raw_input()
        plt.close()