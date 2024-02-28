import numpy as np
import torch

def l1_proximal(Y, lam):
	# estimate the solution of argmin 1/2|x-y|^{2} + lam * |x|
	return torch.sign(Y) * torch.relu(torch.abs(Y)-lam)
def tv_proximal(Y, lam, D, invrD, iterations =20, rho = 1.0):
	
	# estimate the soluton of argmin 1/2|x-y|^{2} + lam * TV(x)
	N = Y.shape[0]
	d = Y.shape[1]
	X0 = torch.randn(N, d)
	Z0 = torch.randn(N, d-1)
	U0 = torch.randn(N, d-1)

	# perform ADMM iterations
	for i in range(iterations):
		X1 = torch.matmul(Y + torch.matmul(rho * Z0 - U0, D), invrD)
		Z1 = l1_proximal(torch.matmul(X1, D.T) + 1/rho * U0, lam/rho)
		U1 = U0 + rho *(torch.matmul(X1, D.T)-Z1)
		X0 = X1
		Z0 = Z1
		U0 = U1
	return X1