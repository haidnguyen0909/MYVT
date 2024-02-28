import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import pandas as pd
import scipy
import scipy.special
import matplotlib.pyplot as plt
from prox_operators import *


# define generator network
class GeneratorNet(nn.Module):
	def __init__(self, lr=1e-5, input_dim = 10):
		super().__init__()
		self.input_dim = input_dim
		self.fc1_dim = input_dim
		self.fc2_dim = 100
		self.fc3_dim = 100
		self.fc4_dim = 100
		self.fc5_dim = input_dim

		self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
		self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
		self.fc3 = nn.Linear(self.fc2_dim, self.fc2_dim)
		self.fc4 = nn.Linear(self.fc2_dim, self.fc2_dim)
		self.fc5 = nn.Linear(self.fc2_dim, self.fc5_dim)

		self.bn1 = nn.BatchNorm1d(self.fc2_dim)
		self.bn2 = nn.BatchNorm1d(self.fc2_dim)
		self.bn3 = nn.BatchNorm1d(self.fc2_dim)
		self.bn4 = nn.BatchNorm1d(self.fc2_dim)

		self.fc1.weight.data.normal_(0,0.1)
		self.fc2.weight.data.normal_(0,0.1)
		self.fc3.weight.data.normal_(0,0.1)
		self.fc4.weight.data.normal_(0,0.1)
		self.fc5.weight.data.normal_(0,0.1)

		nn.init.constant_(self.fc1.bias, 0)
		nn.init.constant_(self.fc2.bias, 0)
		nn.init.constant_(self.fc3.bias, 0)
		nn.init.constant_(self.fc4.bias, 0)
		nn.init.constant_(self.fc5.bias, 0)

		self.lr = lr
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, input):
		X = input
		X = self.fc1(X)
		X = torch.tanh(self.bn1(X))
		X = self.fc3(X)
		X = torch.tanh(self.bn3(X))
		X = self.fc4(X)
		X = torch.tanh(self.bn4(X))
		X = self.fc5(X)
		return X


class DualNet(nn.Module):
	def __init__(self, lr=1e-4, input_dim = 100):
		super().__init__()
		self.fc1_dim = input_dim
		self.fc2_dim = 100
		self.fc3_dim = 1

		self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
		self.fc2 = nn.Linear(self.fc2_dim, self.fc3_dim)


		self.fc1.weight.data.normal_(0,0.1)
		self.fc2.weight.data.normal_(0,0.1)

		nn.init.constant_(self.fc1.bias, 0)
		nn.init.constant_(self.fc2.bias, 0)

		self.lr = lr
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, input):
		X = self.fc1(input)
		X = torch.tanh(X)
		X = self.fc2(X)
		return X





def Grad_KLDivergence(target, theta, Y, G_net, H_net, grad_iters = 2):
	batch_size = theta.shape[0]
	for i in range(grad_iters):
		theta_ = theta.detach()
		theta_.requires_grad = True
		term1 = H_net(theta_).mean()
		term2 = torch.log(torch.exp(H_net(Y)).mean())
		loss = -term1 + term2
		H_net.optimizer.zero_grad()
		autograd.backward(loss)
		H_net.optimizer.step()

	theta_ = theta.detach().clone()
	theta_.requires_grad = True
	output = H_net(theta_)
	theta_grad = autograd.grad(outputs=output, inputs=theta_, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
	return theta_grad



def TVDenosing_KLDivergence(target, truth, Y, batch_size, G_net, H_net, alpha1, alpha2, lam, num_iters, grad_iters=1):
	lr = G_net. lr
	print("Running experiment for TV denoising task with KL Divergence, Regularization parameter Alpha_1 = ", alpha1)
	d = Y.shape[1]


	# define D and its inversion for total variation semi-norm
	D = np.zeros((d-1, d))
	for i in range(d-2):
		D[i, i] = -1.0
		D[i, i+1] = 1.0

	I = torch.eye(d)
	rho = 1.0
	D = torch.tensor(D).float()
	invrD = torch.linalg.inv(I + rho * torch.matmul(D.T, D)).float()

	theta_new = None

	alpha = 0.0
	for iteration in range(num_iters):
		if alpha < alpha1:
			alpha += 0.00001 # increasing alpha over iterations
		eps = torch.FloatTensor(batch_size, G_net.input_dim).uniform_(0.0, 1.0) # gaussian noise
		theta = G_net.forward(eps) # let noise go through the generator to generate samples
		delta = Grad_KLDivergence(target, theta, Y, G_net, H_net, grad_iters = grad_iters)
		theta_ = theta.detach().clone()

		prox_delta = 1/lam*(theta_ - tv_proximal(theta_, lam, D, invrD, iterations =20, rho=rho))
		theta_new = theta_ - lr*delta - lr*alpha2*theta_ - lr * alpha*prox_delta
		theta_new_ = theta_new.detach().clone()
		G_net.optimizer.zero_grad()
		autograd.backward(-theta, grad_tensors = (theta_new_ - theta_)/lr)
		G_net.optimizer.step()

		if iteration % 100==0:
			tv_semi_norm = torch.sum(torch.abs(torch.matmul(theta_new, D.T))).detach().numpy()/theta_new.shape[0]
			potential = target.potential(theta).mean().detach().numpy()
			print("Iteration ", iteration, " : potential ", potential, "   TV semi-norm (smoothness) ", tv_semi_norm)
	return theta_new

























