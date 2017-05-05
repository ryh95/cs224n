#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    # default is l2 norm
    norm =  np.linalg.norm(x,axis=1)[:,None]
    x = x/norm
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE

    # predicted->Vc(column vector)
    # outputVectors->U.T (one row is one token,U corresponding to assignment1 word2vec section)
    # target->o

    # cost->J
    # gradPred->J w.r.t Vc
    # grad->J w.r.t U.T

    # my solution
    y_hat = softmax(np.matmul(outputVectors,predicted).flatten()) # notice to flatten because you should softmax on a vector not a matrix
    y = np.zeros(y_hat.shape)
    y[target] = 1

    cost = -np.log(y_hat[target])
    gradPred = np.matmul(outputVectors.T,y_hat-y)
    grad = np.matmul((y_hat-y).reshape(-1,1),predicted.reshape(1,-1))

    # Kangwei Ling's solution
    # reference : https://github.com/kevinkwl/cs224n/blob/master/assignment1/q3_word2vec.py
    # y_ = softmax(np.dot(outputVectors, predicted).flatten())
    #
    # cost = - np.log(y_[target])
    #
    # delta = y_
    # delta[target] -= 1
    #
    # gradPred = np.dot(outputVectors.T, delta)
    # grad = np.dot(delta.reshape(-1, 1), predicted.T)

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE

    # parameter explanation see softmaxCostAndGradient function

    # my solution

    # first_ = -np.asscalar(np.log(sigmoid(np.dot(outputVectors[target].reshape(1,-1),predicted))))
    #
    # second_ = 0
    # for idx in indices:
    #     if idx != target:
    #         second_ += np.asscalar(np.log(sigmoid(-np.matmul(outputVectors[idx].reshape(1,-1),predicted))))
    #
    # cost = first_ - second_ # this should be a number
    #
    #
    # first_ = (sigmoid(np.matmul(outputVectors[target].reshape(1,-1),predicted))-1)*outputVectors[target]
    #
    # second_ = 0
    # for idx in indices:
    #     if idx != target:
    #         second_ += (sigmoid(-np.matmul(outputVectors[idx].reshape(1,-1),predicted))-1)*outputVectors[idx]
    #
    # gradPred = (first_ - second_).T # this shape should be equal to predicted
    #
    #
    # grad = np.zeros(outputVectors.shape)
    # for idx in indices:
    #     if idx != target:
    #         # pay attention to +=
    #         grad[idx] += (-(sigmoid(-np.matmul(outputVectors[idx].reshape(1, -1), predicted)) - 1) * predicted).reshape(-1,)
    #     elif idx == target:
    #         grad[idx] = (sigmoid(np.matmul(outputVectors[idx].reshape(1,-1),predicted))-1)*predicted.reshape(-1,)


    # Kangwei Ling's solution
    # reference: https://github.com/kevinkwl/cs224n/blob/master/assignment1/q3_word2vec.py

    output = outputVectors[indices, :]
    sign = - np.ones(len(indices)).reshape(-1, 1)
    sign[0] = 1

    product = sigmoid(np.dot(output, predicted) * sign)
    product_1 = (product - 1) * sign

    cost = - np.log(product).sum()

    gradPred = np.dot(output.T, product_1.reshape(-1, 1))
    grad = np.zeros(outputVectors.shape)

    # the negative sample may have the save index
    gradmixed = np.dot(product_1.reshape(-1, 1), predicted.reshape(1, -1))

    for i in xrange(len(indices)):
        grad[indices[i], :] += gradmixed[i, :]


    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE


    # Kangwei Ling's solution
    # reference : https://github.com/kevinkwl/cs224n/blob/master/assignment1/q3_word2vec.py

    # center_idx = tokens[currentWord]
    # predicted = inputVectors[center_idx].reshape(-1, 1)  # make it a column vector
    #
    # target_indices = [tokens[word] for word in contextWords]
    # result = [word2vecCostAndGradient(predicted, idx, outputVectors, dataset) for idx in target_indices]
    # costs, gradIns, gradOuts = zip(*result)
    #
    # cost = np.sum(costs)
    # # print np.sum(gradIns, axis=0).shape
    # gradIn[center_idx] = np.sum(gradIns, axis=0).reshape(1, -1)  # get back to row forms
    #
    # gradOut = np.sum(gradOuts, axis=0)

    # cost2 = 0.0
    # gradIn2 = np.zeros(inputVectors.shape)
    # gradOut2 = np.zeros(outputVectors.shape)


    # my solution
    predicted = inputVectors[tokens[currentWord]]
    # make it column vector
    predicted = predicted.reshape(-1,1)

    for string in contextWords:
        target = tokens[string]
        cost_, gradPred, grad = word2vecCostAndGradient(predicted,target,outputVectors,dataset)
        cost += cost_
        gradIn[tokens[currentWord]] += gradPred.reshape(-1,)
        gradOut += grad

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()