import numpy as np

from layers import *

class SVM(object):
	"""
	A binary SVM classifier with optional hidden layers.

	Note that this class does not implement gradient descent; instead, it
	will interact with a separate Solver object that is responsible for running
	optimization.

	The learnable parameters of the model are stored in the dictionary
	self.params that maps parameter names to numpy arrays.
	"""

	def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: An integer giving the size of the input
		- hidden_dim: An integer giving the size of the hidden layer
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		"""
		self.params = {}
		self.reg = reg

		############################################################################
		# TODO: Initialize the weights and biases of the model. Weights            #
		# should be initialized from a Gaussian with standard deviation equal to   #
		# weight_scale, and biases should be initialized to zero. All weights and  #
		# biases should be stored in the dictionary self.params, with first layer  #
		# weights and biases using the keys 'W1' and 'b1' and second layer weights #
		# and biases (if any) using the keys 'W2' and 'b2'.                        #
		############################################################################
		if hidden_dim:
			self.params['W1'] = np.random.normal(scale = weight_scale, size = ((input_dim, hidden_dim)))
			self.params['b1'] = np.zeros(hidden_dim)
			self.params['W2'] = np.random.normal(scale = weight_scale, size = ((hidden_dim, )))
			self.params['b2'] = np.zeros(1)
		else:
			self.params['W1'] = np.random.normal(scale = weight_scale, size = ((input_dim, ))) #weight_scale * np.random.randn(input_dim, 1)
			self.params['b1'] = np.zeros(1)
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################


	def loss(self, X, y=None):
		"""
		Compute loss and gradient for a minibatch of data.

		Inputs:
		- X: Array of input data of shape (N, D)
		- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

		Returns:
		If y is None, then run a test-time forward pass of the model and return:
		- scores: Array of shape (N,) where scores[i] represents the classification 
		score for X[i].

		If y is not None, then run a training-time forward and backward pass and
		return a tuple of:
		- loss: Scalar value giving the loss
		- grads: Dictionary with the same keys as self.params, mapping parameter
		  names to gradients of the loss with respect to those parameters.
		"""  
		scores = None
		############################################################################
		# TODO: Implement the forward pass for the model, computing the            #
		# scores for X and storing them in the scores variable.                    #
		############################################################################
		c1, c_relu, c2 = None, None, None
		cache1, cache_relu, cache2 = None, None, None
		W1, b1, W2, b2 = self.params['W1'], self.params['b1'], None, None 
		if 'W2' in self.params:
			W2, b2 = self.params['W2'], self.params['b2']
			c1, cache1 = fc_forward(X, W1, b1)
			c_relu, cache_relu = relu_forward(c1)
			c2, cache2 = fc_forward(c_relu, W2, b2)
			scores = c2
			#print(X.shape, c1.shape, c2.shape, W1.shape, b1.shape, W2.shape, b2.shape)
		else:
			c1, cache1 = fc_forward(X, W1, b1)
			scores = c1
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		# If y is None then we are in test mode so just return scores
		if y is None:
		  return scores

		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the model. Store the loss          #
		# in the loss variable and gradients in the grads dictionary. Compute data #
		# loss and make sure that grads[k] holds the gradients for self.params[k]. #
		# Don't forget to add L2 regularization.                                   #
		#                                                                          #
		############################################################################

		loss, dout = svm_loss(scores, y)
		loss += self.reg * np.sum(W1**2)
		if 'W2' in self.params:
			loss += self.reg * np.sum(W2**2)
			dc_relu, dw2, db2 = fc_backward(dout, cache2)
			grads['b2'] = db2
			grads['W2'] = dw2 + self.reg * W2
			dc1 = relu_backward(dc_relu, cache_relu)
			dx, dw1, db1 = fc_backward(dc1, cache1)
			grads['b1'] = db1
			grads['W1'] = dw1 + self.reg * W1
			#print(dx.shape, dc1.shape, dout.shape, dw1.shape, db1.shape, dw2.shape, db2.shape )
			
		else:
			dc, dw, db = fc_backward(dout, cache1)
			grads['W1'] = dw + self.reg * W1
			grads['b1'] = db
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads
