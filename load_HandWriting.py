#load_HandWriting
import numpy as np
def load_HandWriting():
    file_X = open('X_data.txt', 'r')
    file_Y = open('Y_data.txt', 'r')
    
    #X=np.array([])
    #Y=np.array([])
    
    num_x = sum(1 for line in file_X)
    num_y = sum(1 for line in file_Y)
    file_Y.seek(0)
    file_X.seek(0)
    print num_x
    print num_y

    num_data = 5000
    num_feature = 400
#    print file_Y.read().split()[3]
    y_str = (file_Y.read().split())
    X_str = (file_X.read().split())
    print len(X_str)
    X = np.array([float(X_str[k]) for k in range(num_data*num_feature)]).reshape(num_data, num_feature)
    Y = np.array([float(y_str[k]) for k in range(num_data)])
    
    return X, Y
