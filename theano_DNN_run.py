from network3NickReduction import Network
from network3NickReduction import load_data_shared
from network3NickReduction import FullyConnectedLayer
from theano_sigmoid import sigmoid

training_data,validation_data, test_data = load_data_shared(5000,1000,1000)

layer12 = FullyConnectedLayer(69,128,sigmoid)
layer23 = FullyConnectedLayer(128,48,sigmoid)

layer_list = []
layer_list.append(layer12)
layer_list.append(layer23)

# make a class
DNN1 = Network(layer_list,1000)
