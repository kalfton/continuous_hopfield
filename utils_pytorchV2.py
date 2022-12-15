import numpy as np
from numpy import random
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from cmath import inf
from scipy import stats
import math
from hopfield_class_torch import Hopfield_network

def make_pattern(n_pattern, n_neuron, perc_active = 0.5):
    patterns = torch.from_numpy(random.rand(n_pattern, n_neuron)).type(torch.float32)
    patterns[patterns>1-perc_active]=1-1e-5
    patterns[patterns<=1-perc_active]=1e-5 # when the state_g is equal to 0 or 1, state_x will go to infinity
    return patterns

def train_equalibium_prop(network:Hopfield_network, patterns:torch.tensor, lr=0.01, dt_train=0.1, max_loop_train=100, gamma = 100):
    
    n_pattern = patterns.shape[0]
    lossfun = nn.MSELoss()
    training_loss = []

    torch.set_grad_enabled(False)
    original_dt = network.dt
    network.dt = dt_train
    original_max_loop = network.max_loop
    network.max_loop = max_loop_train

    n_loop=0
    while  n_loop<=original_max_loop:
        losses = torch.zeros(n_pattern)
        for i in range(n_pattern):
            rho1, converge = network.evolve(patterns[i])
            rho2, converge = network.evolve(rho1, patterns[i], gamma = gamma)

            W = network.W()+lr*(rho2[:,None]@rho2[None, :]-rho1[:,None]@rho1[None, :])
            W = W-torch.diag(torch.diag(W))
            network.set_W(W)
            network.set_b(network.b()+lr*(rho2-rho1))
            losses[i] = lossfun(patterns[i], rho1)
        loss = torch.mean(losses)
        
        if loss<network.min_error:
            break

        if n_loop%100==0:
            print(f'Step: {n_loop}, Loss: {loss}')
            training_loss.append(loss.detach().numpy().item())
        
        n_loop+=1
    if n_loop>=original_max_loop:
        success = False
    else:
        success = True

    network.dt = original_dt
    network.max_loop = original_max_loop
    torch.set_grad_enabled(True)

    plt.figure()
    plt.plot(np.arange(0, n_loop, 100), training_loss)
    plt.show(block = False)
    

    return network, success


def train_PLA(network:Hopfield_network, patterns:torch.tensor, lr=0.01, k1 = 0.5, k2 = 2):
    n_pattern = patterns.shape[0]

    torch.set_grad_enabled(False)
    n_loop=1
    while  n_loop<=network.max_loop:
        check_array = torch.zeros(n_pattern).type(torch.bool)
        for i in range(n_pattern):
            P = patterns[i]
            aa = k1*torch.sqrt(torch.sum(network.W()**2, axis=1))+k2*network.beta
            bb = (network.W()@P+network.b())*(2*P-1)
            epsilon = torch.heaviside(aa-bb + 1e-15 ,torch.tensor([1.0])) # plus the 1e-15 to prevent the computer calculation error, and when aa-bb=0, the loop still need to run.

            delta_W = lr*epsilon[:,None]*(2*P[:,None]-1)*P
            delta_W = delta_W+delta_W.transpose(0,1)
            delta_b = lr*epsilon*(2*P-1)

            W = network.W()+delta_W
            network.set_W(W-torch.diag(torch.diag(W)))
            network.set_b(network.b()+delta_b)

            if torch.all(epsilon==0):
                check_array[i]=True
        
        if torch.all(check_array):
            break

        if n_loop%100==0:
            print(f'{n_loop} of loops done, out of {network.max_loop}')
        
        n_loop+=1
    if n_loop>=network.max_loop:
        success = False
    else:
        success = True

    torch.set_grad_enabled(True)

    return network, success

def train_back_prop(network:Hopfield_network, patterns, lr=0.01, n_step = 2, dt_train = 0.5):
    # Move the network to GPU if possible:
    if torch.cuda.is_available():
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:  
        dev = "cpu"
    dev="cpu"
    device = torch.device(dev)

    network = network.to(device)
    patterns = patterns.to(device)
    
    #BPTT Training: 
    n_pattern = patterns.shape[0]

    optimizer = optim.Adam(network.parameters(), lr=lr)
    lossfun = nn.MSELoss()


    training_loss = []
    n_loop = 0
    while n_loop<network.max_loop:
        losses = torch.zeros(n_pattern)
        for i in range(n_pattern):
            P = patterns[i]
            optimizer.zero_grad()
            retrieve_P = network(P, n_step, dt_train)
            losses[i] = lossfun(P, retrieve_P)
        loss = torch.mean(losses)
        if loss<network.min_error:
            break
        loss.backward()

        if torch.any(torch.isnan(network.h2h.weight.grad)):
            print('pause')

        optimizer.step()

        if n_loop%100==0:
            print(f'Step: {n_loop}, Loss: {loss}')
            training_loss.append(loss.detach().numpy().item())

        #For debugging: 
        if torch.isnan(network.h2h.weight.min()):
            print('pause')

        n_loop+=1

    if n_loop>=network.max_loop:
        success = False
    else:
        success = True

    plt.figure()
    plt.plot(np.arange(0, n_loop, 100), training_loss.to(torch.device("cpu")))
    plt.show(block = False)
    network = network.to(torch.device("cpu"))

    return network, success


def compare_rho_permutation_test(X1:np.ndarray, Y1:np.ndarray, X2:np.ndarray, Y2:np.ndarray, nperm = 5000):
    # input are two pairs of data, and we compare their correlation 
    # fix the # permutations

    size1 = X1.shape[0]
    size2 = X2.shape[0]

    # set a void vector for the dif of correl.
    corr_diff = np.zeros(nperm)

    # now start permuting
    for i in range(nperm):
        # sample an index
        if size1==size2:
            idx1 = random.choice(size1, size1, replace=True)
            idx2 = idx1
        else:
            idx1 = random.choice(size1, size1, replace=True)
            idx2 = random.choice(size2, size2, replace=True)

        # calculate the permuted correlation in the first condition
        (corr1, p) = stats.spearmanr(X1[idx1], Y1[idx1])

        # calculate the permuted correlation in the second condition
        (corr2, p) = stats.spearmanr(X2[idx2], Y2[idx2])

        # store the dif. of correlations
        corr_diff[i] = corr1-corr2

    
    # compute the Monte Carlo approximation of the permutation p-value
    if np.any((corr_diff>0) | (corr_diff<0)):
        p = 2*min(np.mean(corr_diff>0), np.mean(corr_diff<0))
    else:
        p = np.nan
    return p
