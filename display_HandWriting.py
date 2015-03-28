import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
def display_HandWriting(X):
    row = 1
    column = 1
    sample = random.sample(range(5000), row*column)
    
    
    f, axarr = plt.subplots(row, column)
    fig = np.zeros([20,20])
    for r in range(row):
        for c in range(column):
            sample_data = X[sample[r*c+c]][:].reshape(20,20)
            """
            for i in range(20):
                for j in range(20):
                    fig[i][j] = sample_data[19-i][j]
            fig[::-1]
            """
            #axarr[r][c].imshow((sample_data), cmap=cm.Greys_r)
            axarr.imshow((sample_data), cmap=cm.Greys_r)
    plt.show()
    return 