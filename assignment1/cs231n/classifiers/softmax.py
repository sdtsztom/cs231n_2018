import numpy as np
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
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
 
  for i in range(num_train):
    scores = X[i].dot(W)
	  
    # shift values for 'scores' for numeric reasons (over-flow cautious)
	# see http://cs231n.github.io/linear-classify/#softmax
    scores -= scores.max()
    scores_exp_sum=np.sum(np.exp(scores))
    crs = np.exp(scores[y[i]])/scores_exp_sum
    loss -= np.log(crs)

    # grad
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += (crs-1) * X[i] # for correct class
      else:
        dW[:, j] += np.exp(scores[j])/scores_exp_sum * X[i]     # for incorrect classes

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
 
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # loss
  # score: N by C matrix containing class scores
  scores = X.dot(W)
  scores -= np.max(scores,axis=1,keepdims=True)
  scores_exp = np.exp(scores)
  scores_exp_sums = np.sum(scores_exp, axis=1)
  crs = scores_exp[range(num_train), y]/scores_exp_sums
  loss -= np.sum(np.log(crs))/num_train 
  loss += reg * np.sum(W * W)

  # grad
  s = scores_exp/scores_exp_sums[:,np.newaxis]
  s[range(num_train), y] -= 1
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

