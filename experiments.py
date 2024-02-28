import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy
import scipy.special
import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng()


from targets import *
from networks import *


# generate synthetic data
d = 100 # dimensionality
N = 500 # No. samples

D = np.zeros((d-1, d)) # for total variation semi-norm computation
for i in range(d-2):
	D[i, i] = -1.0
	D[i, i+1] = 1.0


# generate truth signal X
X = np.random.randn(N, d)  
X[:,0:10] = 1.0 
X[:,10:20] = np.linspace(2,3,10, endpoint=True) 
X[:,20:30] = 1.0 
X[:,30:40] = np.linspace(3,2,10, endpoint=True)
X[:,40:50] = 1.0
X[:,50:60] = np.linspace(1,0,10, endpoint=True)
X[:,60:70] = 1.0
X[:,70:80] = 0.0
X[:,80:90] = 1.0
X[:,90:d] = np.linspace(0,1,10, endpoint=True)


# generate noisy signals Y
eps = np.random.randn(N, d)*0.2
Y =  eps + X
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()


# convert data to tensor
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
D = torch.tensor(D).float()

# visualize some generated signals
index = [i for i in range(d)]
plt.rcParams["figure.figsize"] = [5.0, 3.0]
plt.rcParams["figure.autolayout"] = True
plt.figure(1)
plt.plot(index, X[0,:], color='red',label = "clean signal", linestyle="-")
plt.plot(index, Y[0,:], color='green',label = "noisy signal 1", linestyle="-")
plt.plot(index, Y[1,:], color='blue', label = "noisy signal 2", linestyle="-")
plt.plot(index, Y[2,:], color='orange',label = "noisy signal 3", linestyle="-")
plt.legend()
plt.xlim(0, 100)
plt.ylim(-2,3.5)
plt.savefig("generatedsignal.png")




# prepare parameters for experiments
lr = 0.0005# for KL


target = TVTarget(X)
G_net = GeneratorNet(lr, d)
H_net = DualNet(lr, d)

# run experiment for signal demoising with TV semi-norm and KL divergence
denoisedsamples = TVDenosing_KLDivergence(target, X, Y, N, G_net, H_net, alpha1=0.01, alpha2=0.00001, lam=0.000001, num_iters=2000, grad_iters=2)


# visualized denoised generated samples
denoise_1 = denoisedsamples[0].detach().numpy().tolist()
denoise_2 = denoisedsamples[1].detach().numpy().tolist()
denoise_3 = denoisedsamples[2].detach().numpy().tolist()
index = [i for i in range(d)]
		
plt.figure(2)
plt.plot(index, denoise_1, color="green",label = "sample 1", linestyle="-")
plt.plot(index, denoise_2, color="blue", label = "sample 2", linestyle="-")
plt.plot(index, denoise_3, color="orange", label = "sample 3", linestyle="-")
plt.rcParams["figure.figsize"] = [5.0, 3.0]
plt.rcParams["figure.autolayout"] = True
plt.legend()
plt.ylim(-2,3.5)
plt.savefig("denoised.png")


