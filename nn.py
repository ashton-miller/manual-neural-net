"""This program will build a neural network by hand in python"""
import math
import numpy as np

def sigmoid(x_input):
    """This function calculates the sigmoid activation function"""
    return 1 / (1 + math.exp(-x_input))
def sigmoid_derivative(x_input):
    """This function is the derivative of sigmoid"""
    return sigmoid(x_input) * (1 - sigmoid(x_input))

class NeuralNetwork:
    """ Class to define and initialize the neural network"""
    def __init__(self, x, y):
        self.input = x
        self.weights_1 = np.random.rand(self.input.shape[1], 4)
        self.weights_2 = np.random.rand(4, 1)
        self.y_y = y
        self.output = np.zeros(self.y_y.shape)
        self.layer_1 = sigmoid(np.dot(self.input, self.weights_1))

    def feed_forward(self):
        """calculate the predicted output from the current weights """
        self.layer_1 = sigmoid(np.dot(self.input, self.weights_1))
        self.output = sigmoid(np.dot(self.layer_1, self.weights_2))
    def backprop(self):
        """use chain rule to find derivative of the loss function"""
        d_weights2 = np.dot(
            self.layer_1.T, (2*(self.y_y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(
            self.input.T, (
                np.dot(2*(self.y_y - self.output) * sigmoid_derivative(
                    self.output), self.weights_2.T)* sigmoid_derivative(self.layer_1)))


        self.weights_1 = d_weights1
        self.weights_2 = d_weights2
