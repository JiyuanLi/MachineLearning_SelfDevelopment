# ===================================================================
# Author: Li, Jiyuan
# The script tries to show the basic method of Support Vector Machine
# SMO without kernel function is the chosen learning algorithm in
# this scrip. The script also indicates where the kernel function
# should be used if it is required
# ===================================================================
import numpy as np


class SupportVectorMachine:
    def __init__(self, c=6, tol=0.001, miter=300):
        self.X = np.mat([0])  # This needs to be transferred into Matrix
        self.Y = np.mat([0])  # This needs to be transferred into Matrix
        self.C = c
        self.tol = tol
        self.miter = miter
        self.support_vector = []
        self.alpha = np.mat([0])  # This needs to be transferred into Matrix
        self.b = float(0)
        self.E = []  # This needs to be transferred into Matrix; Ei = Ui-Yi
        self.recognition_results = []

    def load_training_data(self, train_samples, expected_results):
        self.X = np.mat(train_samples)  # shape: [No_Samples, No_Dimensions]
        self.Y = np.mat(expected_results).T  # Shape: [No_Samples, 1], Labels should be either 1 or -1
        self.recognition_results = []
        row_num_samples = self.X.shape[0]
        row_num_results = self.Y.shape[0]
        if row_num_samples != row_num_results:
            print "Wrong Training Data: Number of Samples is not matched with number of expected results!"
            return 1
        self.alpha = np.mat(np.zeros((row_num_samples, 1), float))  # Shape: [No_Samples, 1]
        self.E = []  # Shape will be: [No_Samples, 1]
        for i in range(0, row_num_samples):
            self.E.append(float(np.multiply(self.alpha, self.Y).T * self.X * self.X[i, :].T) + self.b - float(self.Y[i]))  # Kernel function if required

    def update_alpha(self, i, j, e_j):
        # -------------------Used For Debug------------------#
        # print "Alpha Update Begins for element {} and {}".format(i, j)
        # ---------------------------------------------------#
        if i == j:
            return 0

        # Get required variables for element i
        alpha_i = float(self.alpha[i].copy())
        x_i = self.X[i, :]  # x_i is the feature vector of sample X[i]
        y_i = self.Y[i]  # y_i is the expected recognition result for sample X[i]
        e_i = float(np.multiply(self.alpha, self.Y).T * self.X * x_i.T) + self.b - float(y_i)  # Kernel function if required

        # Get required variables for element j
        alpha_j = float(self.alpha[j].copy())
        x_j = self.X[j, :]  # x_j is the feature vector of sample X[j]
        y_j = self.Y[j]  # y_j is the expected recognition result for sample X[j]

        s = float(y_i*y_j)

        # Calculate L and H
        if y_i == y_j:
            L = float(max(0, alpha_i + alpha_j - self.C))
            H = float(min(self.C, alpha_i + alpha_j))
        else:
            L = float(max(0, alpha_j - alpha_i))
            H = float(min(self.C, self.C + alpha_j - alpha_i))

        # -------------------Used For Debug------------------#
        # print "L: {}".format(L)
        # print "H: {}".format(H)
        # ---------------------------------------------------#

        if L == H:
            return 0

        # Calculate K11, K12, K22, use Kernel Function if required
        eta = float(x_i * x_i.T + x_j * x_j.T - 2.0 * x_i * x_j.T)

        # Update alpha_j
        if eta > 0:
            alpha_j_new = alpha_j + float(y_j * (e_i - e_j) / eta)
            if (alpha_j_new < L) and (L < H):
                alpha_j_new = L
            elif alpha_j_new > H:
                alpha_j_new = H
        else:
            print "eta <= 0!"  # ---------------------This is different from the paper----------------------------------
            return 0

        # Update alpha_i
        if abs(alpha_j_new - alpha_j) < 0.00001:  # -----------------This is different from the paper-------------------
            # print "Ignore small update"
            return 0
        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        # -------------------Used For Debug------------------#
        # print "s: {}".format(s)
        # print "alpha_i: {}".format(alpha_i)
        # print "alpha_j: {}".format(alpha_j)
        # print "alpha_i_new: {}".format(alpha_i_new)
        # print "alpha_j_new: {}".format(alpha_j_new)
        # ---------------------------------------------------#

        # Update threshold b: Kernel Function may needed
        b1 = float(self.b - (e_i + y_i * (alpha_i_new - alpha_i) * x_i * x_i.T +
                   y_j * (alpha_j_new - alpha_j) * x_i * x_j.T))
        b2 = float(self.b - (e_j + y_i * (alpha_i_new - alpha_i) * x_i * x_j.T +
                   y_j * (alpha_j_new - alpha_j) * x_j * x_j.T))

        if (0 < alpha_i_new) and (self.C > alpha_i_new):
            self.b = b1
        elif (0 < alpha_j_new) and (self.C > alpha_j_new):
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        # -------------------Used For Debug------------------#
        # print "b_new: {}".format(self.b)
        # ---------------------------------------------------#

        # Update Error Cache
        self.E[i] = e_i
        self.E[j] = e_j

        # Store updated Alpha into self.alpha[]
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        return 1

    def examine_example(self, j):
        alpha_j = self.alpha[j].copy()
        x_j = self.X[j, :]  # x_j is the feature vector of sample X[j]
        y_j = self.Y[j]  # y_j is the expected recognition result for sample X[j]
        e_j = float(np.multiply(self.alpha, self.Y).T * self.X * x_j.T) + self.b - float(y_j)  # Kernel function if required
        r_j = e_j * y_j  # r_j = u_j * y_j - y_j^2 = u_j * y_j - 1
        if (r_j < -self.tol and alpha_j < self.C) or (r_j > self.tol and alpha_j > 0):  # Derived from KKT Condition
            # i selection method 1: choice heuristic
            num_non_0_or_c_alpha = 0
            for alpha_test in self.alpha:
                if alpha_test != 0 and alpha_test != self.C:
                    num_non_0_or_c_alpha += 1
            if num_non_0_or_c_alpha > 1:
                if e_j > 0:
                    sorted_index = np.argsort(self.E)
                    i = sorted_index[0] if sorted_index[0] != j else sorted_index[1]
                else:
                    sorted_index = np.argsort(np.multiply(self.E, -1))
                    i = sorted_index[0] if sorted_index[0] != j else sorted_index[1]
                if self.update_alpha(i, j, e_j):
                    return 1
            # i selection method 2: Should start from an random position
            for i in range(0, self.alpha.shape[0]):
                alpha_test = self.alpha[i]
                if alpha_test != 0 and alpha_test != self.C and i != j:
                    if self.update_alpha(i, j, e_j):
                        return 1
            # i selection method 3: Should start from an random position
            for i in range(0, self.alpha.shape[0]):
                if i != j:
                    if self.update_alpha(i, j, e_j):
                        return 1
        return 0

    def start_training(self):
        if self.alpha.shape[0] < 3:
            print "Not enough training samples! Please use load_training_data function to load more samples"
            return 0
        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all == 1:
            num_changed = 0
            if examine_all == 1:
                for j in range(0, self.alpha.shape[0]):
                    num_changed += self.examine_example(j)
            else:
                for j in range(0, self.alpha.shape[0]):
                    alpha_test = self.alpha[j]
                    if alpha_test != 0 and alpha_test != self.C:
                        num_changed += self.examine_example(j)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

    def recognize(self, input_data):
        input_data_mat = np.mat(input_data)
        sample_number = input_data_mat.shape[0]
        sample_dimension = input_data_mat.shape[1]
        if sample_dimension != self.X.shape[1]:
            print r"Dimension of Input sample is not matched! Feature Dimension should be A x {}".format(self.X.shape[1])
        for i in range(0, sample_number):
            test_sample = input_data_mat[i, :]
            single_recognition_result = float(np.multiply(self.alpha, self.Y).T * self.X * test_sample.T) + self.b
            self.recognition_results.append(single_recognition_result)

if __name__ == '__main__':
    train_set = [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]
    labels = [1, 1, 1, -1, -1, -1]
    SVM = SupportVectorMachine()
    SVM.load_training_data(train_set, labels)
    SVM.start_training()
    SVM.recognize([2, 0])
    print SVM.alpha
    print SVM.b
    print SVM.recognition_results
