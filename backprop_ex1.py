from neural_net import *
from pprint import pprint

nn = Nerual_Net([1, 2, 1])
nn.debug = True

train = np.array([[0.13000, 0.90000], [0.42000, 0.23000]])

# nn.rand_weights()
Theta1 = np.array([[0.40000, 0.10000], [0.30000, 0.20000]])
Theta2 = np.array([0.70000, 0.50000, 0.60000])
nn.weights_list.append(Theta1)
nn.weights_list.append(Theta2)
print("weights_list: \n", nn.weights_list, "\n")
print("--------------------------------------------\n")
print("Computing the error/cost, J, of the network")
cost = nn.accumulate_cost(train)
print("Total J: {}".format(cost))
print("--------------------------------------------\n")
print("Running backpropagation")

