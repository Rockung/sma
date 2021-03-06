import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # here, it is easy to run into numeric instability. Don't forget the       #
  # regularization!                                         #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
     scores = X[i].dot(W)
     scores -= max(scores)                    # prevent overflow    
     scores = np.exp(scores) / np.sum(np.exp(scores))  # compute probabilities for classes
     loss += -np.log(scores[y[i]])               # the loss is -log p of the correct class
     scores[y[i]] -= 1                       # dw = (p-1) * xi if j == yi
     for j in xrange(num_classes):
      dW[:,j] += scores[j] * X[i]               # dw = p * xi if j != yi

  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  dW = dW/num_train + reg* W 
  #############################################################################
  #                          END OF YOUR CODE              #
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful    #
  # here, it is easy to run into numeric instability. Don't forget the      #
  # regularization!                                        #
  #############################################################################
  num_train = X.shape[0]  
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1, 1)                 # prevent overflow
  scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1) # compute the probabilities along the classes

  # loss
  loss = -np.sum(np.log(scores[range(num_train), list(y)]))          # sum the loss for the correct classes
  loss /= num_train                                     # average for a image
  loss +=  0.5* reg * np.sum(W * W)                         # regularization

  # gradient
  scores[range(num_train), list(y)] -= 1 # p-1 for the correct classes
  dW = (X.T).dot(scores)
  dW = dW / num_train + reg* W 
  #############################################################################
  #                          END OF YOUR CODE               #
  #############################################################################

  return loss, dW

