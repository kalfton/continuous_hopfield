import numpy as np
from numpy import random
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from cmath import inf
from scipy import stats
import scipy
import math
from matplotlib import cm
import pandas as pd
from utils_pytorchV2 import *
import warnings

retr_max_allow=0.01

def similarity_measurement(pattern1, pattern2):
    assert pattern1.shape[-1] == pattern2.shape[-1], "Two pattern's dimensions do not equal"
    assert pattern1.ndim<3 and pattern2.ndim<3, "Can't calculate patterns with dimension >=3"
    n_neuron = pattern1.shape[-1]
    if pattern2.ndim==2:
        pattern2 = pattern2.transpose()

    similarity = np.dot(pattern1*2-1, pattern2*2-1)/n_neuron
    return similarity
    
def L1_norm_dist(pattern1, pattern2):
    assert pattern1.shape[-1] == pattern2.shape[-1], "Two pattern's dimensions do not equal"
    assert pattern1.ndim<3 and pattern2.ndim<3, "Can't calculate patterns with dimension >=3"
    dist = np.linalg.norm(pattern1-pattern2, ord=1, axis=-1)
    return dist

def create_init_pattern(target_patterns, n_init_pattern, perc_active, reinit_ratio, pattern_init, neuron_ids=[]):
    # randomly initialize the pattern around the target stored patterns
    # make the neurons which included in neuron_ids as 1

    # target_patterns: n_pattern * n_neuron np array: All the target patterns to be/has stored in a network.
    # n_init_pattern: number of initial patterns to be generated
    
    n_neuron = target_patterns.shape[1]
    n_pattern = target_patterns.shape[0]

    target_pattern_ind = np.zeros(n_init_pattern)
    target_pattern_ind = target_pattern_ind.astype(int)
    init_patterns = np.zeros([n_init_pattern, n_neuron])*np.nan

    if pattern_init==3: # init the pattern around the stored patterns and pick the one that is closer to the target pattern than other stored patterns.
        n=0
        while n<n_init_pattern:
            tag_ind = random.randint(n_pattern)
            targ_pattern = target_patterns[tag_ind,:]
            new_pattern = targ_pattern.copy()

            # resample the pattern from the stored pattern
            reinit_num = int(reinit_ratio*n_neuron)
            new_subpattern = random.rand(reinit_num)
            new_subpattern[new_subpattern>1-perc_active]=act_state
            new_subpattern[new_subpattern<=1-perc_active]=inact_state
            
            inds = random.choice(range(n_neuron),reinit_num, replace=False)
            new_pattern[inds] = new_subpattern

            new_pattern[neuron_ids] = act_state

            # check whether the init pattern is closer to the target pattern than the other stored patterns.
            if np.abs(similarity_measurement(new_pattern, target_patterns).max() - similarity_measurement(new_pattern, targ_pattern))<1e-6:
                if not np.nanmax(L1_norm_dist(new_pattern, init_patterns))<retr_max_allow: # prevent sampling the same state twice
                    init_patterns[n,:] = new_pattern
                    target_pattern_ind[n] = tag_ind
                    n+=1

        # # sanity check: the init pattern are close to the target pattern
        # max_similarity = np.matmul(init_patterns*2-1, (target_patterns*2-1).transpose()).max(axis=1)
        # target_similarity = np.diag(np.dot(init_patterns*2-1, (target_patterns[target_pattern_ind,:]*2-1).transpose()))
        # assert np.all(np.abs(max_similarity - target_similarity)<1e-6)

    elif pattern_init==4: # Ring initialization

        reinit_local = reinit_ratio/(2*perc_active*(1-perc_active)) 
        reinit_local = np.min([reinit_local,1])
        ring_similarity = 1-reinit_ratio

        n=0
        while n<n_init_pattern:
            tag_ind = random.randint(n_pattern)
            targ_pattern = target_patterns[tag_ind,:]
            new_pattern = targ_pattern.copy()

            # resample the pattern from the stored pattern
            reinit_num = int(reinit_ratio*n_neuron)
            new_subpattern = random.rand(reinit_num)
            new_subpattern[new_subpattern>1-perc_active]=act_state
            new_subpattern[new_subpattern<=1-perc_active]=inact_state
            
            inds = random.choice(range(n_neuron),reinit_num, replace=False)
            new_pattern[inds] = new_subpattern

            new_pattern[neuron_ids] = act_state

            # check whether the init pattern is closer to the target pattern than the other stored patterns.
            if np.abs(similarity_measurement(new_pattern, target_patterns).max() - similarity_measurement(new_pattern, targ_pattern))<1e-6\
                and similarity_measurement(new_pattern, targ_pattern)>ring_similarity-0.03 and similarity_measurement(new_pattern, targ_pattern)<ring_similarity+0.03:
                if not np.nanmax(L1_norm_dist(new_pattern, init_patterns))<retr_max_allow: # prevent sampling the same state twice
                    init_patterns[n,:] = new_pattern
                    target_pattern_ind[n] = tag_ind
                    n+=1

        # # sanity check: the init pattern are close to the target pattern
        # max_similarity = np.matmul(init_patterns*2-1, (target_patterns*2-1).transpose()).max(axis=1)
        # target_similarity = np.diag(np.dot(init_patterns*2-1, (target_patterns[target_pattern_ind,:]*2-1).transpose()))
        # assert np.all(max_similarity == target_similarity)

    else:
        raise Exception("Invalid pattern_init value")

    #Kaining temporarily make it:
    init_patterns[init_patterns>0.5] = 1-1e-5
    init_patterns[init_patterns<=0.5] = 1e-5

    return init_patterns, target_pattern_ind

def make_scatter_plot(ax, xdata:dict, ydata:dict, color_data:dict = None, color_bin_n=1):
    # x_data, y_data, color_data: Dict structs which contain the name of the data and the data itself

    n_bin = color_bin_n
    cmap = cm.get_cmap('cool',n_bin)
    plt.sca(ax)
    xlabel = list(xdata.keys())[0]
    ylabel = list(ydata.keys())[0]
    if color_data is None:
        plt.scatter(xdata[xlabel], ydata[ylabel], s=12)
    else:
        color_label = list(color_data.keys())[0]
        max_c_value = np.nanmax(color_data[color_label])
        min_c_value = np.nanmin(color_data[color_label])
        if max_c_value>min_c_value:
            bin_labels = np.arange(min_c_value, max_c_value-1e-10, (max_c_value-min_c_value)/n_bin)+(max_c_value-min_c_value)/n_bin/2
        else:
            bin_labels = np.ones(n_bin)
        category_result = np.array(pd.cut(color_data[color_label], n_bin, labels = bin_labels, ordered=False)) 

        plt.scatter(xdata[xlabel], ydata[ylabel], s=12, c=category_result, cmap=cmap, vmax = max_c_value, vmin = min_c_value)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(color_label, rotation=270)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Some statistics:
    (rho, p) = stats.spearmanr(xdata[xlabel], ydata[ylabel])
    plt.title(f"rho_all = {rho:.3f}, p = {p:.3f}")

def retrieval_results(network:Hopfield_network, init_patterns_torch: torch.Tensor, target_pattern_ind, patterns, normalized_grad = False):
    if normalized_grad:
        retrieved_patterns, success, retrieval_time = network.evolve_batch_normalized(init_patterns_torch, use_tau=True)
    else:
        retrieved_patterns, success, retrieval_time = network.evolve_batch(init_patterns_torch)

    retrieved_patterns = retrieved_patterns.detach().numpy()
    success = success.detach().numpy()
    retrieval_time = retrieval_time.detach().numpy()
    n_init_pattern = init_patterns_torch.size()[0]

    converge_category = np.zeros(n_init_pattern)
    for i in range(converge_category.size):
        if (L1_norm_dist(patterns,retrieved_patterns[i]).min() > retr_max_allow):
            converge_category[i] = 1 # retrieved spurious patterns
        elif L1_norm_dist(patterns[target_pattern_ind[i]],retrieved_patterns[i]) < retr_max_allow:
            converge_category[i] = 0 # retrieved target pattern
        else:
            converge_category[i] = 2 # retrieved another stored pattern

    if np.any(~success):
        warnings.warn("Warning: Some initial patterns have not converge")
    return retrieval_time, converge_category, retrieved_patterns

def func_for_optim(t_set, network, patterns, init_patterns, target_pattern_ind):
    normalizer = 1/(np.mean(1/t_set))
    t_set = t_set/normalizer
    network.set_tau(t_set)
    retrieved_patterns, success, retrieval_time = network.evolve_batch(torch.from_numpy(init_patterns).float())
    retrieved_patterns = retrieved_patterns.detach().numpy()
    success = success.detach().numpy()
    retrieval_time = retrieval_time.detach().numpy()
    n_init_pattern = init_patterns.shape[0]

    success_converge = L1_norm_dist(patterns[target_pattern_ind,:],retrieved_patterns) < retr_max_allow

    per_success_retrieval = np.sum(success_converge)/n_init_pattern
    average_retrieval_time = np.mean(retrieval_time)
    average_retrieval_time_success = np.mean(retrieval_time[success_converge==0])
    return average_retrieval_time_success


# scipy.integrate.solve_ivp TODO: convert np array to tensor
# This is for scipy.integrate.solve_ivp
def dynamics(t, state_x:np.ndarray, network):
    if type(state_x)==np.ndarray:
        state_x = torch.from_numpy(state_x).to(torch.float32)
    state_g = network.g(state_x, network.beta)
    #dynamics = network.g_derivative(state_x, network.beta)*(network.h2h(state_g)-state_x)
    dynamics = (network.h2h(state_g)-state_x)
    return dynamics.detach().numpy()
    
def stop_event(t, state_x:np.ndarray, network):
    state_x_torch = torch.from_numpy(state_x).to(torch.float32)
    state_g = network.g_inverse(state_x_torch, network.beta)
    new_state_x, new_state_g = network.recurrence(state_x_torch, state_g, dt=0.01)
    #return torch.max(torch.abs(new_state_x-state_x_torch))-network.min_error
    return 1000-t

def event(t, y,z):
    return y[0] - 2
event.terminal = True


# stop_event.terminal=True
    

def evolve_odesolver(state_g_torch:torch.Tensor, network:Hopfield_network):
    t_span = np.array([0,20000], dtype='float32')
    n_init_pattern = state_g_torch.shape[0]
    n_neuron = state_g_torch.shape[1]

    retrieved_pattern = np.zeros([n_init_pattern, n_neuron])
    converge = np.zeros(n_init_pattern).astype('bool')
    retrieval_time = np.zeros(n_init_pattern)
    
    for i in range(n_init_pattern):
        state_x_0 = network.g_inverse(state_g_torch[i], network.beta).numpy()
        solution = scipy.integrate.solve_ivp(dynamics ,t_span, state_x_0, method = 'RK45', event= (stop_event,event), args=(network,))
        if solution.y_events is None:
            retrieved_pattern[i,:] = solution.y[:,-1]
            converge[i] = False
            retrieval_time[i] = solution.t[-1]
        else:
            retrieved_pattern[i,:] = solution.y_events[0]
            converge[i] = True
            retrieval_time[i] = solution.t_events[0]

        # for debug & visualization:
        plt.figure(); plt.plot(solution.t, solution.y.T); plt.show()
        print(f"iter {i}")

    retrieved_pattern = network.g(retrieved_pattern, network.beta).numpy

    return retrieved_pattern, converge, retrieval_time
    
    
    

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
