from neural_net import *

nn = Nerual_Net([2, 4, 3, 2])
nn.debug = True

# Training instance 1
# 		x: [0.32000   0.68000]
# 		y: [0.75000   0.98000]
# 	Training instance 2
# 		x: [0.83000   0.02000]
# 		y: [0.75000   0.28000]
train = np.array([[[0.32, 0.68], [0.75, 0.98]], 
        [[0.83, 0.02], [0.75, 0.28]]])

Theta1 = np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])
Theta2 = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]])
Theta3 = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])

nn.weights_list.append(Theta1)
nn.weights_list.append(Theta2)
nn.weights_list.append(Theta3) 

print("weights_list: \n", nn.weights_list, "\n")
print("--------------------------------------------\n")
print("Computing the error/cost, J, of the network")
cost = nn.accumulate_cost(train)
print("Total J: {}".format(cost))
print("--------------------------------------------\n")
print("Running backpropagation")
nn.back_propagate(train)