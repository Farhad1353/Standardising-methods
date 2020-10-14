import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

def sample_polynomial_data(m=20, order=3, _range=1):
    coeffs = np.random.randn(order + 1) # initialise random coefficients for each order of the input + a constant offset
    print(Polynomial(coeffs))
    poly_func = np.vectorize(Polynomial(coeffs)) # 
    X = np.random.randn(m)
    X = np.random.uniform(low=-_range, high=_range, size=(m,))
    Y = poly_func(X)
    return X, Y, coeffs #returns X (the input), Y (labels) and coefficients for each power

def train(num_epochs, X, Y, H):
    for e in range(num_epochs): # for this many complete runs through the dataset
        y_hat = H(X) # make predictions
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) # calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw # compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb # compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) # update model weight and bias

def plot_h_vs_y(X, y_hat, Y):
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    plt.scatter(X, y_hat, c='b', label='Prediction', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

class MultiVariableLinearHypothesis:
    def __init__(self, n_features):
        self.n_features = n_features
        self.b = np.random.randn() ## initialise bias
        self.w = np.random.randn(n_features) ## initialise weights
        
    def __call__(self, X): # what happens when we call our model, input is of shape (n_examples, n_features)
        y_hat = np.matmul(X, self.w) + self.b ## make prediction, now using vector of weights rather than a single value
        return y_hat # output is of shape (n_examples, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
    
    def calc_deriv(self, X, y_hat, labels):
        diffs = y_hat -labels
        dLdw = 2 * np.array([
            np.sum(diffs * X[:, i]) / m 
            for i in range(self.n_features)
        ]) ## calculate 
        dLdb = 2 * np.sum(diffs) / m
        return dLdw, dLdb

def create_polynomial_inputs(X, order=3):
    new_dataset = np.array([np.power(X, i) for i in range(1, order + 1)]).T ## add powers of the original feature to the design matrix
    return new_dataset # new_dataset should be shape [m, order]

def standardize_data(dataset):
    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0) ## get mean and standard deviation of dataset
    standardized_dataset  = (dataset-mean)/std
    return standardized_dataset

m=100
num_epochs = 200
learning_rate = 0.1
highest_order_power = 4
####### Here we can define our Features(X) and Labels(Y) differntly to suit the model we like to train ################ 
X, Y, ground_truth_coeffs = sample_polynomial_data(m, 1, _range=10)
polynomial_augmented_inputs = create_polynomial_inputs(X, highest_order_power) ## need normalization to put higher coefficient variables on the same order of magnitude as the others
H = MultiVariableLinearHypothesis(n_features=highest_order_power) ## initialise multivariate regression model
train(num_epochs, polynomial_augmented_inputs, Y, H) ## train model
plot_h_vs_y(X, H(polynomial_augmented_inputs), Y)

learning_rate = 0.01
highest_order_power = 20

X_polynomial_augmented = create_polynomial_inputs(X, highest_order_power) ## create poly inputs
X_standardized = standardize_data(X_polynomial_augmented) ## standardise
H = MultiVariableLinearHypothesis(n_features=highest_order_power) ## init model
print(X_standardized)
train(num_epochs, X_standardized, Y, H) ## train model
plot_h_vs_y(X, H(X_standardized), Y) # plot hypothesis vs labels