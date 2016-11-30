# ====================================================================
# Author: Li, Jiyuan
# This script tries to show the basic method of Linear Regression
# One assumption to make this script work is that the features of
# input dataset have only 1 dimension, the algorithm then will map
# this 1 dimensional feature to higher dimensional space (default 3)
# Regularization technique is not used in this method, but should be
# simple to be added [Add Lambda/(2M)*SUM(Theta^2) into cost function]
# ====================================================================
import numpy as np


class LinearRegression:
    def __init__(self, dimension=3, alpha=0.01):
        self.dimension = dimension
        self.alpha = alpha
        self.theta = np.mat(np.zeros((1, self.dimension), float))

    def get_feature_set(self, element):
        feature_list = list()
        for i in range(0, self.dimension):
            feature_list.append(pow(element, i))
        return np.mat(feature_list)

    def training(self, training_set, training_target):
        m = len(training_set)  # m is the number of training samples
        tmp_theta = self.theta
        training_feature_set = list()
        for i in range(0, m):
            tmp_feature = self.get_feature_set(training_set[i])
            training_feature_set.append(tmp_feature.getA()[0])
        training_feature_mat = np.mat(training_feature_set)
        for i in range(0, 10000):
            print "=========Training========="
            for j in range(0, self.dimension):
                cost = (np.array(self.recognize(training_set))-np.array(training_target))*training_feature_mat[:, j]
                tmp_theta[0, j] -= self.alpha/m*cost[0, 0]
                print "Cost: {}".format(str(cost[0, 0]))
                print "Update: {}".format(str(self.alpha/m*cost[0, 0]))
                print "tmp_theta[0, j]: {}".format(str(tmp_theta[0, j]))
            self.theta = tmp_theta
            print "*******Updated theta******"
            print self.theta
            print "=========================="
        return 0

    # Input argument should be a list and the length is the number of samples
    def recognize(self, test_sample):
        hypothesis = list()
        for element in test_sample:
            feature_set = self.get_feature_set(element)
            h_temp = self.theta * feature_set.T
            hypothesis.append(h_temp[0, 0])
        return hypothesis

if __name__ == "__main__":
    x_train = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_train = [1.0, 0.0, 1.0, 4.0, 9.0]

    lr = LinearRegression()
    lr.training(x_train, y_train)

    x_test = [-2.0, -1.0]
    y_test = lr.recognize(x_test)
    print "Final Theta: {}".format(str(lr.theta))
    print "Test Result: {}".format(str(y_test))
