import numpy as np
from numpy import random
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from cmath import inf
from scipy import stats
import math
from hopfield_class import Hopfield_network

def make_pattern(n_pattern, n_neuron, perc_active = 0.5):
    P = random.rand(n_pattern, n_neuron)
    P[P>1-perc_active]=1-1e-5
    P[P<=1-perc_active]=1e-5 # this is to allow continuous hopfield network has so tollerrance
    return P

def train_equalibium_prop(network:Hopfield_network, P, lr=0.01):
    n_pattern = P.shape[0]

    n_loop=1
    while  n_loop<=network.max_loop:
        check_array = np.zeros(n_pattern).astype(bool)
        for i in range(n_pattern):
            rho1, converge = network.evolve(P[i])
            rho2, converge = network.evolve_F(rho1, P[i], gamma = 100)

            network.W = network.W+lr*(rho2[:,None]@rho2[None, :]-rho1[:,None]@rho1[None, :])
            network.b = network.b+lr*(rho2-rho1)
            if np.max(np.abs(rho1-P[i]))<network.min_error:
                check_array[i]=True
        
        if np.all(check_array):
            break

        if n_loop%500==0:
            print(f'{n_loop} of loops done, out of {network.max_loop}')

        #debug
        if np.any(np.isnan(network.W)):
            print('pause')
        
        n_loop+=1
    if n_loop>=network.max_loop:
        success = False
    else:
        success = True

    return network, success


def train_PLA(network:Hopfield_network, P, lr=0.01, k1 = 0.5, k2 = 2):
    n_pattern = P.shape[0]

    n_loop=1
    while  n_loop<=network.max_loop:
        check_array = np.zeros(n_pattern).astype(bool)
        for i in range(n_pattern):
            P_origin = P[i]
            aa = k1*np.sqrt(np.sum(network.W**2, axis=1))+k2*network.beta
            bb = (network.W@P_origin+network.b)*(2*P_origin-1)
            epsilon = np.heaviside(aa-bb + 1e-15 ,1) # plus the 1e-15 to prevent the computer calculation error, and when aa-bb=0, the loop still need to run.

            delta_W = lr*epsilon[:,np.newaxis]*(2*P_origin[:,np.newaxis]-1)*P_origin
            delta_W = delta_W+delta_W.transpose()
            delta_b = lr*epsilon*(2*P_origin-1)

            network.W = network.W+delta_W
            network.W -= np.diag(np.diag(network.W))
            network.b = network.b+delta_b

            if np.all(epsilon==0):
                check_array[i]=True
        
        if np.all(check_array):
            break

        if n_loop%1000==0:
            print(f'{n_loop} of loops done, out of {network.max_loop}')
        
        n_loop+=1
    if n_loop>=network.max_loop:
        success = False
    else:
        success = True

    return network, success

def train_backprop(network:Hopfield_network, P, lr=0.01, k1 = 0.5, k2 = 2):
    n_pattern = P.shape[0]

    batchsize = n_pattern

    # define a torch version network.


def compare_rho_permutation_test(X1, Y1, X2, Y2, nperm = 5000):
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
