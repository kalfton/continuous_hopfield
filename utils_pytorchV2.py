import numpy as np
from numpy import random
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from cmath import inf
from scipy import stats
import math
import time
from hopfield_class_torch import Hopfield_network

act_state = 1-1e-5
inact_state = 1e-5 # when the state_g is equal to 0 or 1, state_x will go to infinity
training_max_iter = 60000

def make_pattern(n_pattern, n_neuron, perc_active = 0.5):
    patterns = torch.from_numpy(random.rand(n_pattern, n_neuron)).type(torch.float32)
    patterns[patterns>1-perc_active]=act_state
    patterns[patterns<=1-perc_active]=inact_state
    return patterns

def retreive_batch_patterns(patterns, network):
    # This function is slow in running speed, use network.evolve_batch(...) instead.

    n_pattern = patterns.shape[0]
    n_neuron = patterns.shape[1]
    stored_patterns = torch.zeros((n_pattern, n_neuron))
    retrieval_time = torch.zeros(n_pattern)
    done = torch.zeros(n_pattern)
    for i in range(n_pattern):
        P = patterns[i]
        stored_patterns[i,:], done[i], retrieval_time[i] = network.evolve(P)
    return stored_patterns, done, retrieval_time

def train_equalibium_prop(network:Hopfield_network, patterns:torch.tensor, lr=0.01, dt_train=0.1, max_loop_train=100, gamma = 1):
    #batch the training set decrease the capacity, why?
    n_pattern = patterns.shape[0]
    lossfun = nn.MSELoss()
    training_loss = []

    torch.set_grad_enabled(False)
    original_dt = network.dt
    network.dt = dt_train
    original_max_loop = network.max_loop
    network.max_loop = max_loop_train

    n_loop=0
    while  n_loop<=training_max_iter:
        # losses = torch.zeros(n_pattern)
        # for i in range(n_pattern):
        #     rho1, converge, _ = network.evolve(patterns[i])
        #     rho2, converge, _ = network.evolve(rho1, patterns[i], gamma = gamma)

        #     W = network.W()+lr*(rho2[:,None]@rho2[None, :]-rho1[:,None]@rho1[None, :])
        #     W = W-torch.diag(torch.diag(W))
        #     # network.set_W(W)
        #     # network.set_b(network.b()+lr*(rho2-rho1))

        #     losses[i] = lossfun(patterns[i], rho1)
        # loss = torch.mean(losses)

        # divide the pattern into minibatch randomly to add stocastic to the training:
        n_mini_batch=2
        batchsize = int(torch.ceil(torch.tensor(n_pattern/n_mini_batch)))
        pattern_set = torch.split(patterns[torch.randperm(n_pattern),:], batchsize, dim=0)

        losses = []
        for batched_patterns in pattern_set:
            rho1_batch, converge, _ = network.evolve_batch(batched_patterns)
            rho2_batch, converge, _ = network.evolve_batch(rho1_batch, batched_patterns, gamma = gamma)

            W = network.W()+lr*(rho2_batch.T@rho2_batch-rho1_batch.T@rho1_batch)
            W = W-torch.diag(torch.diag(W))
            network.set_W(W)
            network.set_b(network.b()+lr*torch.sum(rho2_batch-rho1_batch, dim=0))
            losses.append(lossfun(batched_patterns, rho1_batch))
        loss = torch.mean(torch.tensor(losses))

        network.energy_func(network.g_inverse(patterns[1],network.beta), patterns[1])
        if loss<network.min_error:
            break

        if n_loop%100==0:
            print(f'Step: {n_loop}, Loss: {loss}')
            training_loss.append(loss.detach().numpy().item())
        
        n_loop+=1
    if n_loop>=training_max_iter:
        success = False
    else:
        success = True

    network.dt = original_dt
    network.max_loop = original_max_loop
    torch.set_grad_enabled(True)
    
    # Get the most acurate patterns that is stored in the network:
    stored_patterns, _, _ = network.evolve_batch(patterns)
    if lossfun(stored_patterns, patterns)>network.min_error*2:
        success = False

    # plt.figure()
    # plt.plot(np.arange(0, n_loop, 100), training_loss)
    # plt.show(block = False)

    return network, success, stored_patterns


def train_PLA(network:Hopfield_network, patterns:torch.tensor, lr=0.01, k1 = 0.5, k2 = 2):
    n_pattern = patterns.shape[0]

    torch.set_grad_enabled(False)
    n_loop=1
    while  n_loop<=training_max_iter:
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
            print(f'{n_loop} of loops done, out of {training_max_iter}')
        
        n_loop+=1
    if n_loop>=training_max_iter:
        success = False
    else:
        success = True
    
    # Get the most acurate patterns that is stored in the network:
    stored_patterns, _, _ = network.evolve_batch(patterns)

    # start_time = time.time()
    # stored_patterns2, success, retrieval_time= retreive_batch_patterns(patterns, network)
    # print("--- %s seconds ---" % (time.time() - start_time))


    torch.set_grad_enabled(True)

    return network, success, stored_patterns

def train_back_prop(network:Hopfield_network, patterns, lr=0.01, n_step = 2, dt_train = 0.5):
    # Move the network to GPU if possible:
    if torch.cuda.is_available():
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:  
        dev = "cpu"
    dev = "cpu" # here the cpu is faster than gpu
    device = torch.device(dev)

    network = network.to(device)
    patterns = patterns.to(device)
    
    #BPTT Training: 
    n_pattern = patterns.shape[0]
    n_neuron = patterns.shape[1]
    min_error = network.min_error**1.4 # empirically

    optimizer = optim.Adam(network.parameters(), lr=lr)
    lossfun = nn.MSELoss()


    training_loss = []
    n_loop = 0
    while n_loop<training_max_iter:
        # losses = torch.zeros(n_pattern)
        # optimizer.zero_grad()
        # for i in range(n_pattern):
        #     P = patterns[i]
        #     retrieve_P = network(P, n_step, dt_train)
        #     losses[i] = lossfun(P, retrieve_P)
        # loss = torch.mean(losses)

        optimizer.zero_grad()
        retrieve_patterns = network(patterns, n_step, dt_train)
        loss = lossfun(patterns, retrieve_patterns)

        if loss<min_error:
            break
        loss.backward()

        if torch.any(torch.isnan(network.h2h.weight.grad)):
            print('pause')

        optimizer.step()

        if n_loop%100==0:
            print(f'Step: {n_loop}, Loss: {loss}')
            training_loss.append(loss.detach().cpu().numpy().item())

        n_loop+=1

    if n_loop>=training_max_iter:
        success = False
    else:
        success = True

    network = network.to(torch.device("cpu"))

    # Get the most acurate patterns that is stored in the network:
    stored_patterns, _, _ = network.evolve_batch(patterns)
    if lossfun(stored_patterns, patterns)>network.min_error*2:
        success = False

    # plt.figure()
    # plt.plot(np.arange(0, n_loop, 100), training_loss)
    # plt.show(block = False)
    

    return network, success, stored_patterns

