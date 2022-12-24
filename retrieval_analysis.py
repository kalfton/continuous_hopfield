import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from hopfield_class_torch import Hopfield_network
import torch
import torch.nn as nn
import pickle
import time
import utils_pytorchV2 as utils_torch
import utils_retrieval as utils_retr
import warnings

random.seed(0)
torch.manual_seed(0)
n_neuron = 50
n_pattern = 50
perc_active=0.5

n_init_pattern=1000
reinit_ratio = 0.3

start_time = time.time()

# with open('trained_network_torch_PLA.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network1, patterns) = data_saved

# Generate the storing patterns
patterns = utils_torch.make_pattern(n_pattern, n_neuron, perc_active = perc_active)
# Initiate the network
network1 = Hopfield_network(n_neuron, dt=0.01)
# Train the network:
# network1, success, patterns = utils_torch.train_back(network1, patterns, lr=0.01, k1 = 0.0, k2 = 2)
network1, success, stored_patterns2 = utils_torch.train_back_prop(network1, patterns, lr=0.01, n_step = 2, dt_train = 0.5)

patterns = patterns.detach().numpy()

# analysis1: 1. Prove that {\tau_i} can affect the retreval. 2. See how much can it acheive?

init_patterns, target_pattern_ind = utils_retr.create_init_pattern(patterns, n_init_pattern, perc_active, reinit_ratio, pattern_init=3, neuron_ids=[])

# set the network with different {\tau_i}, and compare the difference.
# Randomly generate {\tau_i} sets:
max_allow = 0.1
n_t_set = 10
t_sets = 0.1+random.rand(n_t_set, n_neuron)*0.9 # prevent tau too small or too big.
#normalize them:
normalizer = 1/np.sqrt(np.mean(1/t_sets**2, axis=1))
t_sets = t_sets/normalizer[:,None]

per_success_retrieval = np.zeros(n_t_set)
per_wrong_retrieval = np.zeros(n_t_set)
per_spurious_retrieval = np.zeros(n_t_set)
average_retrieval_time = np.zeros(n_t_set)
std_retrieval_time = np.zeros(n_t_set)
min_retrieval_time = np.zeros(n_t_set)

init_patterns_torch = torch.from_numpy(init_patterns).float()

for ii in range(n_t_set):
    print(f"tau set {ii} out of {n_t_set}")

    network1.set_tau(t_sets[ii,:])
    retrieved_patterns, success, retrieval_time = network1.evolve_batch(init_patterns_torch)
    retrieved_patterns = retrieved_patterns.detach().numpy()
    success = success.detach().numpy()
    retrieval_time = retrieval_time.detach().numpy()

    # retrieved_patterns = np.zeros((n_init_pattern, n_neuron))
    # for jj in range(n_init_pattern):
    #     retrieved_patterns[jj,:] = network1.evolve(init_patterns_torch[jj,:])[0].detach().numpy() # can I do it in a batch way to make it quicker?

    # TODO Change the similarity_measurement with L1_dist.
    converge_category = np.zeros(n_init_pattern)
    for i in range(converge_category.size):
        if np.all(utils_retr.similarity_measurement(patterns,retrieved_patterns[i])<=1-max_allow):
            converge_category[i] = 1 # retrieved spurious patterns
        elif utils_retr.similarity_measurement(patterns[target_pattern_ind[i]],retrieved_patterns[i])>1-max_allow:
            converge_category[i] = 0 # retrieved target pattern
        else:
            converge_category[i] = 2 # retrieved another stored pattern

    if np.any(~success):
        warnings.warn("Warning: Some initial patterns have not converge")

    per_success_retrieval[ii] = np.sum(converge_category==0)/n_init_pattern
    per_spurious_retrieval[ii] = np.sum(converge_category==1)/n_init_pattern
    per_wrong_retrieval[ii] = np.sum(converge_category==2)/n_init_pattern
    average_retrieval_time[ii] = np.mean(retrieval_time)
    min_retrieval_time[ii] = np.min(retrieval_time)


print("--- %s seconds ---" % (time.time() - start_time))

print("the biggest different in success retrieval rate = {}".format(per_success_retrieval.max()-per_success_retrieval.min()))

print("end")


    

