from re import A
import numpy as np

def sigmoid_function(x): 
        return 1 / (1 + np.exp(-x))

class Nerual_Net:
    def __init__(self, neurons_list):
        self.neurons_list = neurons_list
        self.weights_list = []
        self.debug = False
        self.regurization_lambda = 0
    
    def rand_weights(self):
        # exclude the output layer
        for i in range(0, len(self.neurons_list) - 1):
            # including the bias neuron
            self.weights_list.append(
                np.random.rand(self.neurons_list[i]+1, self.neurons_list[i + 1]))
    
    def accumulate_cost(self, data):
        J = 0
        for i in range(0, len(data)):
            instance = data[i]
            input = np.array(instance[0])
            expected_output = instance[1]
            if self.debug:
                print("Processing training instance #{}".format(i))
            propagated_output = self.propagate(input)
            current_J = -expected_output * np.log(propagated_output) \
                - (1 - expected_output) * np.log(1 - propagated_output)
            J += current_J
            if self.debug:
                print("propagated_output: {}, expected_output: {}".format(propagated_output, expected_output))
                print("current_J: {} \n".format(current_J))
        return J/len(data)
    
    def propagate(self, input_list):
        activation = input_list
        if self.debug:
            print("activation1: {}".format(activation))
        for i in range(2, len(self.neurons_list)+1):
            # include the bias neuron
            activation = np.append(1, activation)
            z = np.dot(self.weights_list[i-2], activation)
            if self.debug:
                print("z{}: ".format(i), z)
            activation = sigmoid_function(z)
            if self.debug:
                print("activation{}: {}".format(i, activation))
        # f(x)
        return activation

    def back_propagate():
        return 0
