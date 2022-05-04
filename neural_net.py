import numpy as np

def sigmoid_function(x): 
        return 1 / (1 + np.exp(-x))

class Nerual_Net:
    def __init__(self, neurons_list):
        self.neurons_list = neurons_list
        self.weights_list = [] # index 0 is for layer 1
        self.debug = False
        self.regurization_lambda = 0
        # layer count including input/output
        self.num_layer = len(neurons_list)
    

    def rand_weights(self):
        # exclude the output layer
        for i in range(0, self.num_layer - 1):
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
            current_J = -np.multiply(expected_output,np.log(propagated_output)) \
                - np.multiply((1 - expected_output), np.log(1 - propagated_output))
            J += current_J
            if self.debug:
                print("propagated_output: {}, expected_output: {}".format(propagated_output, expected_output))
                print("current_J: {} \n".format(current_J))
        return J/len(data)
    

    def propagate(self, input_list):
        self.activation_list = []
        self.z_list = []
        activation = input_list
        if self.debug:
            print("activation1: {}".format(activation))
        for i in range(2, self.num_layer + 1):
            # include the bias neuron
            activation = np.append(1, activation)
            self.activation_list.append(activation)
            z = np.dot(self.weights_list[i-2], activation) # note here index is i-2 since weight list is indexed from 0
            self.z_list.append(z)
            if self.debug:
                print("z{}: ".format(i), z)
            activation = sigmoid_function(z)
            if self.debug:
                print("activation{}: {}".format(i, activation))
        # f(x)
        self.activation_list.append(activation)
        return activation


    def init_delta_list(self):
        self.delta_list = []
        for i in range(0, self.num_layer):
            self.delta_list.append(np.zeros(self.neurons_list[i]))

    def init_gradient_list(self):
        self.gradient_list = []
        for i in range(0, self.num_layer):
            self.gradient_list.append(np.zeros(self.neurons_list[i]))

    def back_propagate(self, data):
        # self.init_gradient_list()

        for i in range(0, len(data)):
            instance = data[i]
            input = np.array(instance[0])
            expected_output = instance[1]
            if self.debug:
                print("Processing training instance #{}".format(i))
            propagated_output = self.propagate(input)
            self.init_delta_list()
            self.delta_list[self.num_layer - 1] = propagated_output - expected_output
            
            # exclude the input layer (and the output layer)
            for k in range(self.num_layer - 2, 0, -1):
                self.delta_list[k] = np.dot(self.weights_list[k].T, self.delta_list[k + 1]) * self.activation_list[k] * (1 - self.activation_list[k])
                self.delta_list[k] = self.delta_list[k][1:]
            if self.debug:
                print("delta_list: ", self.delta_list)
            
            self.current_gradients = []
            for k in range(0, self.num_layer - 1):
                self.current_gradients.append([])

            for k in range(self.num_layer - 2, -1, -1):
                self.current_gradients[k] = np.array([self.delta_list[k+1]]).T * self.activation_list[k]
            
            if self.debug:
                print("gradient_list: ", self.current_gradients, "\n")

            # cumulate the gradients
            if i == 0:
                self.gradient_list = self.current_gradients
            else:
                for it in range(0, len(self.current_gradients)):
                    self.gradient_list[it] += self.current_gradients[it]
                
        for it in range(0, len(self.gradient_list)):
            self.gradient_list[it] /= len(data)
        
        if self.debug:
            print("total gradient_list: ", self.gradient_list, "\n")

        return 0
