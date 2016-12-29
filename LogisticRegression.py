# ====================================================================
# Author: Li, Jiyuan
# This script tries to show the basic method of Logistic Regression
# One assumption to make this script work is that the desired labels
# are either 1 or 0
# Regularization technique is not used in this method, but should be
# simple to be added [Add Lambda/(2M)*SUM(Theta^2) into cost function]
# ====================================================================
import numpy as np


class LogisticRegression:
    def __init__(self, dimension=2, alpha=0.01):
        self.dimension = dimension
        self.alpha = alpha
        self.theta = 0

    def training(self, training_set, training_target):
        m = len(training_set)                   # m is the number of training samples
        self.dimension = len(training_set[0]) + 1                      # Don't forget theta0: x0 is 1
        self.theta = np.mat(np.zeros((1, self.dimension), float))      # Don't forget theta0: x0 is 1
        updated_training_set = list()
        for element in training_set:
            updated_training_set.append([1] + element)
        training_feature_mat = np.mat(updated_training_set)
        tmp_theta = self.theta

        # Start Training
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

    # Input argument should follow this format: [No_of_Samples, No_of_Dimensions]
    def recognize(self, test_samples):
        hypothesis = list()
        for element in test_samples:
            feature_set = np.mat([1] + element)  # Don't forget theta0: x0 is 1
            h_temp = 1.0 / (1.0 + np.e ** (-self.theta * feature_set.T)[0, 0])
            hypothesis.append(h_temp)
        return hypothesis

if __name__ == "__main__":
    train_set = [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]
    labels = [1, 1, 1, 0, 0, 0]

    lr = LogisticRegression()
    lr.training(train_set, labels)

    y_test = lr.recognize([[5, 0]])
    print "Final Theta: {}".format(str(lr.theta))
    print "Test Result: {}".format(str(y_test))
