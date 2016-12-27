# ====================================================================
# Author: Li, Jiyuan
# This script tries to show the basic method of BP Network Algorithm
# One assumption is the activation function is sigmoid function
# Regularization technique is also considered in this method
# ====================================================================
import numpy as np


class BPNetwork:

    # Define the number of input nodes, hidden layer No., No. of units in 1 hidden layer and No. of output units
    def __init__(self, input_units=3, hidden_layer_number=2, hidden_units_in_one_layer=4, output_units=3):

        self.input_units = input_units
        self.hidden_layer_number = hidden_layer_number
        self.hidden_units_in_one_layer = hidden_units_in_one_layer
        self.output_units = output_units
        self.alpha = 0.3                                # Learning Rate is defined as 0.3
        self.regularization_factor = 0.01               # Regularization Factor is defined as 0.01
        self.init_epsilon = 10.0                        # epsilon used to initialize all parameters
        self.gradient_check_epsilon = 1e-7              # epsilon used to get gradient in normal method
        self.list_of_para_matrix = list()               # List of Theta Matrices for each layer
        self.a = list()                                 # List of Activation Vectors in each layer
        self.a_with_bias = list()                       # List of Activation Vectors and bias unit
        self.delta = list()                             # List of Error Vectors in each layer (Seems to be useless)
        self.list_of_derivative_of_j_theta = list()     # List of derivative of j theta matrices
        self.x_train = list()
        self.y_train = list()
        self.needGradientCheck = True                   # Value used to identify whether gradient check is needed
        self.trainingFinished = False

        # ----------------- Input Layer Parameter Initialization ------------------- #
        # Add Input Vector into a and Error Vector into delta
        self.a.append(np.matrix(np.zeros((self.input_units, 1))))
        self.a_with_bias.append(np.matrix(np.ones((self.input_units+1, 1))))
        self.delta.append(np.matrix(np.zeros((self.input_units, 1))))  # Actually delta[0] is meaningless

        # ----------------- Hidden Layer Parameter Initialization ------------------ #
        # If there is at least 1 hidden layer
        if hidden_layer_number > 0:
            # Add Parameter Matrix of InputLayer <---> 1st HiddenLayer
            tmp_matrix = np.matrix(np.random.random((self.hidden_units_in_one_layer, self.input_units+1)))*2*self.init_epsilon - self.init_epsilon
            self.list_of_para_matrix.append(tmp_matrix)
            self.list_of_derivative_of_j_theta.append(tmp_matrix)

            # Add Parameter Matrix between HiddenLayers, Activation Vector, and Error Vector
            for i in range(0, hidden_layer_number):
                if i < (hidden_layer_number - 1):
                    tmp_matrix = np.matrix(np.random.random((self.hidden_units_in_one_layer, self.hidden_units_in_one_layer+1)))*2*self.init_epsilon - self.init_epsilon
                    self.list_of_para_matrix.append(tmp_matrix)
                    self.list_of_derivative_of_j_theta.append(tmp_matrix)
                self.a.append(np.matrix(np.zeros((self.hidden_units_in_one_layer, 1))))
                self.a_with_bias.append(np.matrix(np.ones((self.hidden_units_in_one_layer+1, 1))))
                self.delta.append(np.matrix(np.zeros((self.hidden_units_in_one_layer, 1))))

            # Add Parameter Matrix between HiddenLayers and OutputLayer
            tmp_matrix = np.matrix(np.random.random((self.output_units, self.hidden_units_in_one_layer+1)))*2*self.init_epsilon - self.init_epsilon
            self.list_of_para_matrix.append(tmp_matrix)
            self.list_of_derivative_of_j_theta.append(tmp_matrix)

        # If No hidden layer is designed
        else:
            tmp_matrix = np.matrix(np.random.random((self.output_units, self.input_units + 1)))*2*self.init_epsilon - self.init_epsilon
            self.list_of_para_matrix.append(tmp_matrix)
            self.list_of_derivative_of_j_theta.append(tmp_matrix)

        # ----------------- Output Layer Parameter Initialization ------------------ #
        # Add Output Vector into a and Error Vector into delta
        self.a.append(np.matrix(np.zeros((self.output_units, 1))))
        self.a_with_bias.append(np.matrix(np.ones((self.output_units, 1))))
        self.delta.append(np.matrix(np.zeros((self.output_units, 1))))

    # Sigmoid function used as activation function for each neuron
    def g(self, z):
        return 1.0/(1.0 + np.exp(-float(z)))

    # Get Output Vector with bias
    def add_bias_element(self, a):
        a_with_bias = np.matrix(np.ones((len(a)+1, 1)))
        for i in range(0, len(a)):
            a_with_bias[i+1] = a[i]
        return a_with_bias

    # Get Output Vector without bias
    def remove_bias_element(self, a):
        a_without_bias = np.matrix(np.ones((len(a) - 1, 1)))
        for i in range(0, len(a_without_bias)):
            a_without_bias[i] = a[i+1]
        return a_without_bias

    # Function used to get final Hypothesis
    def h(self, x_sample, tmp_theta_matrix_list):
        self.a[0] = x_sample
        self.a_with_bias[0] = self.add_bias_element(x_sample)
        for i in range(1, self.hidden_layer_number + 2):
            z = np.dot(tmp_theta_matrix_list[i-1], self.a_with_bias[i-1])
            a_tmp = np.matrix(np.zeros((len(z), 1)))
            for j in range(0, len(z)):
                a_tmp[j, 0] = self.g(z[j, 0])
            self.a[i] = a_tmp
            self.a_with_bias[i] = self.add_bias_element(a_tmp) if i != self.hidden_layer_number + 1 else a_tmp
        return self.a[self.hidden_layer_number + 1]

    # Calculate Cost Function
    def get_j_theta(self, tmp_theta_matrix_list):
        j_theta = 0.0
        regularization_element = 0.0
        for i in range(0, len(self.x_train)):
            x_sample = np.matrix(self.x_train[i]).T
            y_sample = np.matrix(self.y_train[i]).T
            hypothesis = self.h(x_sample, tmp_theta_matrix_list)
            j_theta_sec1 = np.dot(y_sample.T, np.log(hypothesis))
            j_theta_sec2 = np.dot((np.matrix(np.ones((len(y_sample), 1))) - y_sample).T, np.log(np.matrix(np.ones((len(y_sample), 1))) - hypothesis))
            j_theta += (j_theta_sec1 + j_theta_sec2)

        j_theta *= (-1.0/float(len(self.x_train)))
        for i in range(0, len(self.list_of_para_matrix)):
            regularization_element += np.sum(np.power(self.list_of_para_matrix[i], 2))
        regularization_element *= (self.regularization_factor / (2.0 * float(len(self.x_train))))
        j_theta += regularization_element
        return j_theta

    # Calculate derivative of j theta
    def get_list_of_derivative_of_j_theta(self):
        for i in range(0, len(self.list_of_derivative_of_j_theta)):
            self.list_of_derivative_of_j_theta[i] = self.list_of_para_matrix[i] * self.regularization_factor * (-1.0)

        for m in range(0, len(self.x_train)):
            x_sample = np.matrix(self.x_train[m]).T
            y_sample = np.matrix(self.y_train[m]).T

            # Forward Process
            hypothesis = self.h(x_sample, self.list_of_para_matrix)

            # Backward Process
            delta = y_sample - hypothesis
            for n in range(0, self.hidden_layer_number + 1):
                layer_number = self.hidden_layer_number - n
                self.list_of_derivative_of_j_theta[layer_number] += (np.dot(delta, self.a_with_bias[layer_number].T))
                if layer_number == 0:
                    break
                tmp_delta = np.array(np.dot(self.list_of_para_matrix[layer_number].T, delta)) * np.array(self.a_with_bias[layer_number]) * np.array(1.0-self.a_with_bias[layer_number])
                delta = self.remove_bias_element(np.matrix(tmp_delta))

        for n in range(0, self.hidden_layer_number + 1):
            self.list_of_derivative_of_j_theta[n] /= (-1.0*float(len(self.x_train)))

    # Back Propagation Process
    def bp_learning(self):
        # Run 5000 iterations
        for n in range(0, 5000):

            # Get the matrix list of Derivative of J Theta
            self.get_list_of_derivative_of_j_theta()

            # -------------------Used For Debug------------------#
            # Double Check to make sure J theta is decreasing
            # original_j_theta = self.get_j_theta(self.list_of_para_matrix)
            # print "J theta in round {}: {}".format(str(n), str(original_j_theta))
            # ---------------------------------------------------#

            # Gradient Check
            if self.needGradientCheck:
                # Get Initial J Theta
                init_j_theta = self.get_j_theta(self.list_of_para_matrix)
                print "Initial J theta: {}".format(str(init_j_theta))
                # Calculate Gradient in Normal Method
                self.list_of_para_matrix[0][0, 1] += self.gradient_check_epsilon
                tmp_j_theta_1 = self.get_j_theta(self.list_of_para_matrix)
                self.list_of_para_matrix[0][0, 1] -= self.gradient_check_epsilon * 2
                tmp_j_theta_2 = self.get_j_theta(self.list_of_para_matrix)
                tmp_gradient = (tmp_j_theta_1 - tmp_j_theta_2) / (2.0 * self.gradient_check_epsilon)

                # Reset list of para_matrix
                self.list_of_para_matrix[0][0, 1] += self.gradient_check_epsilon

                derived_gradient = self.list_of_derivative_of_j_theta[0][0, 1]
                if abs(tmp_gradient - derived_gradient) < 0.1:
                    print "Gradient Check Pass"
                    self.needGradientCheck = False
                else:
                    print "[Error]Gradient Check Failed!"
                    print "J Theta 1: {}".format(str(tmp_j_theta_1))
                    print "J Theta 2: {}".format(str(tmp_j_theta_2))
                    print "tmp_gradient: {}".format(str(tmp_gradient))
                    print "matrix_gradient: {}".format(str(self.list_of_para_matrix[0][0, 1]))
                    return -1

            # Update Parameters
            for i in range(0, len(self.list_of_para_matrix)):
                self.list_of_para_matrix[i] -= self.alpha * self.list_of_derivative_of_j_theta[i]

        self.trainingFinished = True
        final_j_theta = self.get_j_theta(self.list_of_para_matrix)
        print "BP_Learning Finished"
        print "Final J Theta: {}".format(str(final_j_theta))

    # Entry for training process
    def training(self, x_data, y_data):
        self.x_train = np.matrix(x_data)
        self.y_train = np.matrix(y_data)

        if self.x_train.shape[1] != self.input_units:
            print "[Error] Wrong training data format for X! x_dataset should be of the shape [Samples, Dimensions]"
            return -1
        if self.y_train.shape[0] != self.x_train.shape[0] or self.y_train.shape[1] != self.output_units:
            print "[Error] Wrong training data format for Y! y_dataset should be of the shape [Samples, Classes]"
            return -1

        # Start Training
        self.bp_learning()

    # Evaluation of a test sample
    def recognize(self, x_sample):
        if not self.trainingFinished:
            print "[Error] This network is not trained! Please run 'training(x, y)' before calling this function"
            return -1
        x_sample = np.matrix(x_sample).T
        output = self.h(x_sample, self.list_of_para_matrix)
        return output


if __name__ == "__main__":
    train_set = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]
    labels = [[1], [1], [1], [0], [0], [0]]

    bp_network = BPNetwork(2, 2, 5, 1)
    bp_network.training(train_set, labels)

    y_test = bp_network.recognize([[3.0, 3.0]])
    print "Test Result: "
    print y_test

    y_test = bp_network.recognize([[0.0, 0.0]])
    print "Test Result: "
    print y_test

    y_test = bp_network.recognize([[5.0, 0.0]])
    print "Test Result: "
    print y_test

    y_test = bp_network.recognize([[0.0, 5.0]])
    print "Test Result: "
    print y_test
