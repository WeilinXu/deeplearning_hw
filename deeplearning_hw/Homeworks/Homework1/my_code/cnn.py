import numpy as np

from layers import *


class ConvNet(object):
	"""
	A convolutional network with the following architecture:

	conv - relu - 2x2 max pool - fc - softmax

	You may also consider adding dropout layer or batch normalization layer. 

	The network operates on minibatches of data that have shape (N, C, H, W)
	consisting of N images, each with height H and width W and with C input
	channels.
	"""

	def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
			   hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, addbatchdrop=False,
			   dtype=np.float32):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Size of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
		  of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		############################################################################
		# TODO: Initialize weights and biases for the three-layer convolutional    #
		# network. Weights should be initialized from a Gaussian with standard     #
		# deviation equal to weight_scale; biases should be initialized to zero.   #
		# All weights and biases should be stored in the dictionary self.params.   #
		# Store weights and biases for the convolutional layer using the keys 'W1' #
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
		# of the output affine layer.                                              #
		############################################################################
		
		# conv
		self.params['W1'] = np.random.normal(scale = weight_scale, size = ((num_filters, input_dim[0], filter_size, filter_size)))
		self.addbatchdrop = addbatchdrop

		# pool
		# output2: num_filters, H, W

		# output: 
		self.params['W2'] = np.random.normal(scale = weight_scale, \
			size = ((num_filters*int(input_dim[1]/2)*int(input_dim[2]/2), num_classes)))
		self.params['b2'] = np.zeros(num_classes)
		
		# batch normalization
		if self.addbatchdrop:
			self.params['gamma'] = np.random.normal(scale = weight_scale, size = ((num_classes,)))
			self.params['beta']  = np.random.normal(scale = weight_scale, size = ((num_classes,)))
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)
			
		self.input_size = input_dim
		
	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.

		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		W1 = self.params['W1']
		W2, b2 = self.params['W2'], self.params['b2']
		
		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
		
		# batch normalazation
		bn_param = {'mode': 'train', 'eps': 1e-5, 'momentum': 0.9}
		# dropout
		dropout_param = {'mode': 'train',  'p': 0.5, 'seed': 498}
		
		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
		
		
		
		scores = None
		############################################################################
		# TODO: Implement the forward pass for the three-layer convolutional net,  #
		# computing the class scores for X and storing them in the scores          #
		# variable.                                                                #
		############################################################################
		if self.addbatchdrop:
			gamma = self.params['gamma']
			beta = self.params['beta']
			c, w, h = self.input_size
			X = X.reshape((X.shape[0], c, w, h))
			c1, cache1 = conv_forward2(X, W1, conv_param)
			c_relu, cache_relu = relu_forward(c1)
			c_pool, cache_pool = max_pool_forward(c_relu, pool_param)
			c_drop, cache_drop = dropout_forward(c_pool, dropout_param)
			c2, cache2 = fc_forward(c_drop, W2, b2)
			c_batch, cache_batch = batchnorm_forward(c2, gamma, beta, bn_param)
			scores = c_batch
		else:
			c, w, h = self.input_size
			X = X.reshape((X.shape[0], c, w, h))
			c1, cache1 = conv_forward2(X, W1, conv_param)
			c_relu, cache_relu = relu_forward(c1)
			c_pool, cache_pool = max_pool_forward(c_relu, pool_param)
			c2, cache2 = fc_forward(c_pool, W2, b2)
			scores = c2
		
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################
		
		if y is None:
		  return scores

		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the three-layer convolutional net, #
		# storing the loss and gradients in the loss and grads variables. Compute  #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		############################################################################
		# batch normalazation
		bn_param = {'mode': 'test', 'eps': 1e-5, 'momentum': 0.9}
		# dropout
		dropout_param = {'mode': 'test',  'p': 0.5, 'seed': 498}
		
		if self.addbatchdrop:
			loss, dout = softmax_loss(scores, y)
			loss += self.reg * np.sum(self.params['W1']**2) + self.reg * np.sum(self.params['W2']**2)
			dc_batch, dgamma, dbeta = batchnorm_backward(dout, cache_batch)
			dc2, dw2, b2 = fc_backward(dc_batch, cache2)
			dc_drop = dropout_backward(dc2, cache_drop)
			dc_pool = max_pool_backward(dc_drop, cache_pool)
			dc_relu = relu_backward(dc_pool, cache_relu)
			_, dw1 = conv_backward2(dc_relu, cache1, conv_param)
			
			grads['b2'] = b2
			grads['W2'] = dw2
			grads['W1'] = dw1
			grads['gamma'] = dgamma
			grads['beta'] = dbeta
		else:
			loss, dout = softmax_loss(scores, y)
			loss += self.reg * np.sum(self.params['W1']**2) + self.reg * np.sum(self.params['W2']**2)
			dc2, dw2, b2 = fc_backward(dout, cache2)
			dc_pool = max_pool_backward(dc2, cache_pool)
			dc_relu = relu_backward(dc_pool, cache_relu)
			_, dw1 = conv_backward2(dc_relu, cache1, conv_param)
			
			grads['b2'] = b2
			grads['W2'] = dw2
			grads['W1'] = dw1
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads
  
  
pass
