#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z2 = np.dot(data, W1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W2) + b2
    y_ = softmax(z3)
    cost = - np.log(y_[np.where(labels == 1)]).sum()

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation



    # my solution
    # reference:http://cs231n.github.io/optimization-2/
    # M = len(labels)
    # dz3 = np.zeros((M,Dy))
    # dW2 = np.zeros((M,H,Dy))
    # db2 = np.zeros((M,Dy))
    # da2 = np.zeros((M,H))
    # dz2 = np.zeros((M, H))
    # dW1 = np.zeros((M,Dx,H))
    # db1 = np.zeros((M,H))
    #
    # for i in range(M):
    #     # iterate to compute each row of data
    #     dz3[i] = y_[i] - labels[i] # 1*Dy
    #     dW2[i] = np.dot(a2[i].reshape(-1,1),dz3[i].reshape(1,-1)) # H*Dy
    #     db2[i] = dz3[i] # 1*Dy
    #     da2[i] = np.dot(dz3[i].reshape(1,-1),W2.T) # 1*H
    #     dz2[i] = np.multiply(sigmoid_grad(sigmoid(z2[i])),da2[i]) # 1*H
    #     dW1[i] = np.dot(data[i].reshape(-1,1),dz2[i].reshape(1,-1)) # Dx*H
    #     db1[i] = dz2[i] # 1*H
    #
    # gradW2 = dW2.sum(axis=0)
    # gradb2 = db2.sum(axis=0)
    # gradW1 = dW1.sum(axis=0)
    # gradb1 = db1.sum(axis=0)


    delta2 = y_ - labels
    gradb2 = delta2.sum(axis=0)  # sum over all training samples
    gradW2 = np.dot(np.transpose(a2), delta2)

    gradh = np.dot(delta2, np.transpose(W2))
    delta1 = np.multiply(gradh, np.multiply(a2, 1 - a2))
    gradb1 = delta1.sum(axis=0)
    gradW1 = np.dot(np.transpose(data), delta1)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
