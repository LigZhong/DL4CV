import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i, X_each in enumerate(X):
        scores = np.zeros(len(W[2]))

        for j, W_each in enumerate(W.T):
            z = np.sum(np.multiply(X_each, W_each))
            scores[j] = z

        # Loss
        scores -= np.max(scores) # shift by log C to avoid numerical instability # To Check/.
        exp_scores = np.array([math.exp(k) for k in scores])
        softmax_loss = -1 * np.log(exp_scores[y[i]] / np.sum(exp_scores))
        loss += softmax_loss
        
        # Gradient dW
        dL = exp_scores / np.sum(exp_scores)
        dL[y[i]] -= 1
        dW += np.dot(X_each[:,None], dL[None,:])

        # if i==0:
        #     print('Iteration1  ', z, L2RegLoss, y[i])
        #     print('Raw err  ', scores)
        #     print('Exp err  ', exp_scores)
        #     print('softmax_loss {0:0.2f} = -log( {1:0.2f} / {2:0.2f} )'.
        #           format(softmax_loss, exp_scores[y[i]], np.sum(exp_scores)))
        #     print()

    loss /= len(y)
    loss += 0.5 * reg * np.sum(np.multiply(W, W))
    dW /= len(y)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    y_mat = np.zeros(shape = (X.shape[0], W.shape[1]))
    y_mat[range(len(y)), y] = 1

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Loss
    scores = np.dot(X, W)
    scores -= np.max(scores) # shift by log C to avoid numerical instability
    exp_scores = np.exp(scores)
    softmax_loss = (exp_scores * y_mat).sum(1)
    softmax_loss = softmax_loss / exp_scores.sum(1)
    softmax_loss = - np.log(softmax_loss)
    loss = softmax_loss.sum() / len(y)
    loss += 0.5 * reg * np.sum(np.multiply(W, W))

    # Gradient dW
    dL = exp_scores/exp_scores.sum(1)[:,None]
    dL -= y_mat
    dW = np.dot(X.T, dL)
    dW /= len(y)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

