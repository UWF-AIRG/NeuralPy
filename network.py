import sys
import json
import random
import numpy as np

## Vectorized NN Result - a handy function for getting the output vector of the network.
##		All ten outputs are initialized to 0, but the output representing the answered
##		value is set to 1. So, for example, if the network identifies a '7', the return
##		is [0,0,0,0,0,0,0,1,0,0]
##			0 1 2 3 4 5 6 7 8 9
def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

################## ACTIVATION FUNCTIONS ###################

## Sigmoid - the sigmoid activation function, 1/1+e^(-z)
class Sigmoid:
	## The function itself for feedforward.
	@staticmethod
	def fn(z):
		return 1.0/(1.0+np.exp(-z))
	
	## The derivative for backpropogation.
	@staticmethod
	def prime(z):
		return Sigmoid.fn(z)*(1-Sigmoid.fn(z))

## Tanh - the hyperbolic tangent activation function, (1/2)(tanh(z)+1)
class Tanh:
	## The function itself for feedforward.
	@staticmethod
	def fn(z):
		return (np.tanh(z)+1.0)/2.0
	
	## The derivative for backpropogation.
	@staticmethod
	def prime(z):
		return 0.5*(1.0-(np.tanh(z)**2))

###########################################################

##################### COST FUNCTIONS ######################

## Cross-Entropy - the cross-entropy cost function, sum(-y*ln(a)-(1-y)*ln(1-a))
class CrossEntropyCost:
	## The function itself for calculating the cost.
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	
	## The derivative dC/da for the first step in backpropogation.
	@staticmethod
	def delta(a, y):
		return a-y

## Quadratic - the quadratic cost function, sum((1/2)(a-y)^2)
class QuadraticCost:
	## The function itself for calculating the cost.
	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2
	
	## The derivative dC/da for the first step in backpropogation.
	@staticmethod
	def delta(a, y):
		return a-y

###########################################################

################## WEIGHT INITIALIZERS ####################


## Random weight initializer - assigns each weight a totally random value.
## 		s:		the network
def random_weight_initializer(s):
	s.weights = [np.random.randn(y, x) for x, y in zip(s.sizes[:-1], s.sizes[1:])]

## Smart weight initializer - assigns each weight a random value divided by the square root of how many neurons are in the layer.
##		s:		the network
def smart_weight_initializer(s):
	s.weights = [np.random.randn(y,x)/np.sqrt(x) for x, y in zip(s.sizes[:-1], s.sizes[1:])]

###########################################################

class Network:
	## Network initialization
	##		sizes:			an array of how many neurons are in each layer
	##		cost:			the cost function used (selected from above, default Cross-Entropy)
	##		init:			the weight initialization used (selected from above, default smart)
	##		activation:		the activation function used (selected from above, default sigmoid)
	def __init__(self, sizes, cost=CrossEntropyCost, init=smart_weight_initializer, activation=Sigmoid):
		self.layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.cost = cost
		self.activation = activation
		init(self)	
	
	## Feedforward - the function for putting inputs (a) through the network
	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):																	# for each layer of weights and biases
			a = self.activation.fn(np.dot(w, a)+b)																	# calculate activation for each neuron in the layer
		return a
	
	## Mini Batch Update - the function for updating the network based on a batch of training examples
	##		mini_batch:		the array of examples (x) and their desired output (y)
	##		eta:			the 'learning rate' (already defined by the network)
	##		lmbda:			the regularization parameter (already defined by the network)
	##		n:				the total length of this epoch's training data
	def update_mini_batch(self, mini_batch, eta, lmbda, n):
		nabla_b = [np.zeros(b.shape) for b in self.biases] 															# initialize bias gradient to all 0s
		nabla_w = [np.zeros(w.shape) for w in self.weights] 														# initialize weight gradient to all 0s
		for x, y in mini_batch: 																					# for each training example and expected output
			delta_nabla_b, delta_nabla_w = self.backprop(x, y) 															# backpropogate to get the gradient for this training example
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 												# add this example's bias gradient to the total bias gradient
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] 												# add this example's weight gradient to the total weight gradient
		self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]		# update the weights
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]							# update the biases
	
	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)
		
	def cost_derivative(self, output_activations, y):
		return self.cost.delta(output_activations,y)
	
	## Backpropogation - the method for propogating the error back through the network to update weights and biases.
	##		x:		training example
	##		y:		expected output
	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]															# initialize bias gradient to all 0s
		nabla_w = [np.zeros(w.shape) for w in self.weights]															# initialize weight gradient to all 0s
		activation = x																								# set input as first activation
		activations = [x]																							# array of all activations for calculating weight update (delta*activation)
		zs = []																										# array of all inputs for calculating dActivation/dInput (da/dz)
		for b, w in zip(self.biases, self.weights):																	# for each layer of weights and biases
			z = np.dot(w, activation)+b																					# calculate input (z) for all neurons in layer
			zs.append(z)																								# append layer inputs to array
			activation = self.activation.fn(z)																			# calculate activation (a) for all neurons in layer
			activations.append(activation)																				# append layer activations to array
		delta = self.cost_derivative(activations[-1], y) * self.activation.prime(zs[-1])							# calculate the error, delta, for the output layer (delta = dC/da * Activation'(z))
		nabla_b[-1] = delta																							# set bias gradient to delta (dC/db = delta)
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())													# set weight gradient to delta*previous activations (dC/dw = delta*a^l-1)
		for l in xrange(2, self.layers):																			# for each layer in the network
			z = zs[-l]																									# get inputs for layer
			sp = self.activation.prime(z)																				# get dActivation/dInput at input (Activation'(z))
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp													# calculate error, delta, for layer (delta = dC/da * Activation'(z))
			nabla_b[-l] = delta																							# set bias gradient to delta (dC/db = delta)
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())													# set weight gradient to delta*previous activations (dC/dw = delta*a^l-1)
		return (nabla_b, nabla_w)
	
	## Stochastic Gradient Descent (SGD) - the technical name for the actual training process.
	##		Stochastic: the random initializations and changing of the network.
	##		Gradient: the derivations dC/da for all activations in the network.
	##		Descent: the changing of the weights to minimize the cost - descending the gradient.
	##
	##		training_data: 		the set of all data available to the network
	##		epochs: 			the number of epochs you would like to train for
	##		mini_batch_size:	the number of data used for each batch of an epoch
	##		eta:				the 'learning rate' of the network
	##		lmbda:				the regularization parameter
	def SGD(self, training_data, epochs, mini_batch_size, eta,
			lmbda = 0.0,
			evaluation_data=None,
			monitor_evaluation_cost=False,
			monitor_evaluation_accuracy=False,
			monitor_training_cost=False,
			monitor_training_accuracy=False):
		if evaluation_data: n_data = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(
					mini_batch, eta, lmbda, len(training_data))
			print "Epoch %s training complete" % j
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print "Cost on training data: {}".format(cost)
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				print "Accuracy on training data: {} / {}".format(
					accuracy, n)
			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, lmbda, convert=True)
				evaluation_cost.append(cost)
				print "Cost on evaluation data: {}".format(cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data: {} / {}".format(
					self.accuracy(evaluation_data), n_data)
			print
		return evaluation_cost, evaluation_accuracy, \
			training_cost, training_accuracy
	
	def total_cost(self, data, lmbda, convert=False):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a, y)/len(data)
		cost += 0.5*(lmbda/len(data))*sum(
			np.linalg.norm(w)**2 for w in self.weights)
		return cost
		
	def accuracy(self, data, convert=False):
		if convert:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y))
					   for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in data]
		return sum(int(x == y) for (x, y) in results)
	
	## Save - put this network into a JSON object and save it.
	##		filename:		the file to save to
	def save(self, filename):
		data = {"sizes": self.sizes,
				"weights": [w.tolist() for w in self.weights],
				"biases": [b.tolist() for b in self.biases],
				"cost": str(self.cost.__name__)}
		f = open(filename, "w")
		json.dump(data, f)
		f.close()

## Load - load a network from a file containing a JSON object.
##		filename:		the file to load from
def load(filename):
	f = open(filename, "r")
	data = json.load(f)
	f.close()
	cost = getattr(sys.modules[__name__], data["cost"])
	net = Network(data["sizes"], cost=cost)
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net
