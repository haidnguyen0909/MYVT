import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class TVTarget():
	def __init__(self, X):
		self.X = X
	def potential(self, Y):
		return torch.sum(0.5*(Y-self.X)*(Y-self.X),axis=1)
	def potential_grad(self, Y):
		return (Y - self.X)/(Y.shape[1] * Y.shape[1])