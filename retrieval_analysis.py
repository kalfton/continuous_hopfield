import numpy as np
from numpy import random
from scipy import optimize
from scipy import stats
from matplotlib import pyplot as plt
from hopfield_class_torch import Hopfield_network
import torch
import torch.nn as nn
import pickle
import time
import utils_pytorchV2 as utils_torch
import utils_retrieval as utils_retr
import warnings

random.seed(13)
torch.manual_seed(15)
perc_active=0.5

n_init_pattern=1000
reinit_ratio = 0.2
pattern_init = 4

start_time = time.time()

# with open('data/trained_network_torch_eq_prop_n_50_p_40.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network1, patterns) = data_saved
# patterns = patterns.detach().numpy()

# load stored data:
with open('data/optimized_tau_and_init_states_V1.pickle', 'rb') as f:
    data_saved = pickle.load(f)
tau_opt_1, init_patterns_all, target_pattern_ind_all, network1, patterns = data_saved

network1.max_loop=200000
network1.dt=0.1

# # Generate the storing patterns
# patterns = utils_torch.make_pattern(n_pattern, n_neuron, perc_active = perc_active)
# # Initiate the network
# network1 = Hopfield_network(n_neuron, dt=0.01)
# # Train the network:
# # network1, success, patterns = utils_torch.train_back(network1, patterns, lr=0.01, k1 = 0.0, k2 = 2)
# network1, success, patterns = utils_torch.train_back_prop(network1, patterns, lr=0.01, n_step = 2, dt_train = 0.5)


n_pattern = patterns.shape[0]
n_neuron = patterns.shape[1]


init_patterns_all, target_pattern_ind_all = utils_retr.create_init_pattern(patterns, 2*n_init_pattern, perc_active, reinit_ratio, pattern_init=pattern_init, neuron_ids=[])
init_patterns = init_patterns_all[:n_init_pattern, :]
target_pattern_ind = target_pattern_ind_all[:n_init_pattern]

init_patterns_2 = init_patterns_all[n_init_pattern:, :]
target_pattern_ind_2 = target_pattern_ind_all[n_init_pattern:]

# properties of the network:
bias = network1.b().numpy()
weight = network1.W().numpy()
mean_weight_pos = np.zeros(n_neuron)
mean_weight_neg = np.zeros(n_neuron)
mean_weight = np.zeros(n_neuron)
for i in range(n_neuron):
    include_ind = np.where(weight[i,:]>0)
    mean_weight_pos[i] = np.mean(weight[i,include_ind[0]])
    include_ind = np.where(weight[i,:]<0)
    mean_weight_neg[i] = np.mean(weight[i,include_ind[0]])
    mean_weight[i]  = np.mean(np.abs(weight[i,:]))

pattern_deg = np.sum(patterns>0.5, axis=0)

###############################  Optimization of tau ############################################################
# # find the optimal \tau
# tau_0 = 0.2+ 0.8*(mean_weight.max()-mean_weight)/(mean_weight.max()-mean_weight.min()) #0.2+random.rand(n_neuron)*0.8
# tau_bounds = np.tile(np.array([[0.2, 1]]), (n_neuron,1)) #optimize.Bounds(0.2*np.ones(n_neuron),1*np.ones(n_neuron))

# optimized_result_anneal = optimize.dual_annealing(utils_retr.func_for_optim, tau_bounds, args=(network1, patterns, init_patterns_2, target_pattern_ind_2), \
#      maxfun=10000, x0=tau_0)
# tau_opt_1 = optimized_result_anneal.x

# # save the init states and optimized \tau
# data = [tau_opt_1, init_patterns_all, target_pattern_ind_all, network1, patterns] # save network and original pattern too?
# with open('data/optimized_tau_and_init_states_1.pickle', 'wb') as f:
#     pickle.dump(data, f)

################################# End of optimization ############################################################


# tau_bounds = optimize.Bounds(0.2,1)

# options_NM = {'maxfev': 5000, 'xatol': 0.0001, 'fatol':0.0001}
# optimized_resultNM = optimize.minimize(utils_retr.func_for_optim, tau_0, (network1, patterns, init_patterns_2, target_pattern_ind_2), \
#     bounds = tau_bounds, method ='Nelder-Mead', options=options_NM)
# tau_opt_1 = optimized_resultNM.x
# tau_opt_1 = tau_0




# set the network with different {\tau_i}, and compare the difference.
# Randomly generate {\tau_i} sets:
max_allow = utils_retr.retr_max_allow
n_t_set = 25
# new way of setting tau:
taus = np.sort(0.2+ 0.8*random.rand(n_neuron))
t_sets = np.zeros([n_t_set, n_neuron])
for i in range(n_t_set):
    t_sets[i,:] = random.permutation(taus)

special_tau_name = ['abs weight pos'.rjust(20), 'bias pos'.rjust(20), 'abs weight neg'.rjust(20), 'bias neg'.rjust(20)]
special_tau_name.reverse()
ind_weight = np.argsort(np.argsort(mean_weight))
ind_bias = np.argsort(np.argsort(bias))

t_sets[-1,:] = taus[ind_weight]
t_sets[-2,:] = taus[ind_bias]
t_sets[-3,:] = taus[np.flip(ind_weight)]
t_sets[-4,:] = taus[np.flip(ind_bias)]

# t_sets = 0.2+ 0.8*random.rand(n_t_set, n_neuron) # prevent tau too small or too big.
# special_tau_name = []
#make the last n tau set special:
# special_tau_name = ['abs weight pos'.rjust(20), 'bias pos'.rjust(20), 'abs weight neg'.rjust(20), 'bias neg'.rjust(20), 'constant'.rjust(20), 'optimized train'.rjust(20), 'optimized test'.rjust(20)]
# special_tau_name.reverse()
# t_sets[-1,:] = 0.2+ 0.8*(mean_weight-mean_weight.min())/(mean_weight.max()-mean_weight.min())
# t_sets[-2,:] = 0.2+ 0.8*(bias-bias.min())/(bias.max()-bias.min())
# t_sets[-3,:] = 0.2+ 0.8*(mean_weight.max()-mean_weight)/(mean_weight.max()-mean_weight.min())
# t_sets[-4,:] = 0.2+ 0.8*(bias.max()-bias)/(bias.max()-bias.min())
# t_sets[-5,:] = 0.9*np.ones(n_neuron)
# t_sets[-6,:] = 0.2+ 0.8*(tau_opt_1-tau_opt_1.min())/(tau_opt_1.max()-tau_opt_1.min())
# t_sets[-7,:] = 0.2+ 0.8*(tau_opt_1-tau_opt_1.min())/(tau_opt_1.max()-tau_opt_1.min())

# t_sets[-1,:] = 0.2+ 0.8*(pattern_deg.max()-pattern_deg)/(pattern_deg.max()-pattern_deg.min())
# t_sets[-3,:] = 0.2+ 0.8*(mean_weight_pos.max()-mean_weight_pos)/(mean_weight_pos.max()-mean_weight_pos.min())
# t_sets[-2,:] = 0.2+ 0.8*(mean_weight_neg-mean_weight_neg.min())/(mean_weight_neg.max()-mean_weight_neg.min())


#normalize them:
#normalizer = 1/np.sqrt(np.mean(1/t_sets**2, axis=1))
normalizer = 1/(np.mean(1/t_sets, axis=1))
t_sets = t_sets/normalizer[:,None]

per_success_retrieval = np.zeros(n_t_set)
per_wrong_retrieval = np.zeros(n_t_set)
per_spurious_retrieval = np.zeros(n_t_set)
average_retrieval_time = np.zeros(n_t_set)
std_retrieval_time = np.zeros(n_t_set)
min_retrieval_time = np.zeros(n_t_set)
average_retrieval_time_success = np.zeros(n_t_set)
std_retrieval_time_success = np.zeros(n_t_set)
average_retrieval_time_fail = np.zeros(n_t_set)
std_retrieval_time_fail = np.zeros(n_t_set)
rho_bias_tau = np.zeros(n_t_set)
rho_pos_weight_tau= np.zeros(n_t_set)
rho_neg_weight_tau= np.zeros(n_t_set)

init_patterns_torch = torch.from_numpy(init_patterns).float()
init_patterns_torch_2 = torch.from_numpy(init_patterns_2).float()

for ii in range(n_t_set):
    print(f"tau set {ii} out of {n_t_set}")

    network1.set_tau(t_sets[ii,:])
    if ii == n_t_set-6: # training result
        retrieval_time, converge_category, retrieved_patterns = utils_retr.retrieval_results(network1, init_patterns_torch_2, target_pattern_ind_2, patterns)
    else:
        retrieval_time, converge_category, retrieved_patterns = utils_retr.retrieval_results(network1, init_patterns_torch, target_pattern_ind, patterns)

    per_success_retrieval[ii] = np.sum(converge_category==0)/n_init_pattern
    per_spurious_retrieval[ii] = np.sum(converge_category==1)/n_init_pattern
    per_wrong_retrieval[ii] = np.sum(converge_category==2)/n_init_pattern
    average_retrieval_time[ii] = np.mean(retrieval_time)
    std_retrieval_time[ii] = np.std(retrieval_time)  #/np.sqrt(n_init_pattern)
    min_retrieval_time[ii] = np.min(retrieval_time)
    average_retrieval_time_success[ii] = np.mean(retrieval_time[converge_category==0])
    std_retrieval_time_success[ii] = np.std(retrieval_time[converge_category==0])
    average_retrieval_time_fail[ii] = np.mean(retrieval_time[converge_category!=0])
    std_retrieval_time_fail[ii] = np.std(retrieval_time[converge_category!=0])

    rho_bias_tau[ii] = stats.spearmanr(bias,t_sets[ii,:])[0]
    rho_pos_weight_tau[ii]= stats.spearmanr(mean_weight_pos,t_sets[ii,:])[0]
    rho_neg_weight_tau[ii]= stats.spearmanr(mean_weight_neg,t_sets[ii,:])[0]

# plot the each sets success rate and retrieving time as bar plot.
fig, axes = plt.subplots(3,3, figsize = [19,10])
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)

axes[0,0].bar(range(n_t_set), per_success_retrieval)
axes[0,0].set_ylabel("ratio of successfully retrieval")

axes[0,1].bar(range(n_t_set), average_retrieval_time_success)
axes[0,1].errorbar(range(n_t_set), average_retrieval_time_success, std_retrieval_time_success, None, '.')
axes[0,1].set_ylabel("average time of retrieval (success)")

axes[0,2].bar(range(n_t_set), average_retrieval_time_fail)
axes[0,2].errorbar(range(n_t_set), average_retrieval_time_fail, std_retrieval_time_fail, None, '.')
axes[0,2].set_ylabel("average time of retrieval (fail)")



# set special bar ticks:
axes[0,2].set_xticks(np.arange(len(special_tau_name))+n_t_set-len(special_tau_name))
axes[0,2].set_xticklabels(special_tau_name, rotation=45, ha='right', rotation_mode='anchor')

# axes[1,1].scatter(per_success_retrieval, average_retrieval_time_success)
# axes[1,1].set_ylabel("average time of retrieval (success)")
# axes[1,1].set_xlabel("ratio of successfully retrieval")
# rho, p = stats.spearmanr(per_success_retrieval, average_retrieval_time_success)
# axes[1,1].set_title(f"rho = {rho:.5f}, p = {p:.5f}")

# utils_retr.make_scatter_plot(axes[2,2], {"ratio of successfully retrieval": per_success_retrieval}, \
#     {"average time of retrieval (success)": average_retrieval_time_success}, {"std tau": np.std(t_sets, axis=1)}, color_bin_n=7)

utils_retr.make_scatter_plot(axes[2,2], {"ratio of successfully retrieval": per_success_retrieval}, \
    {"average time of retrieval (success)": average_retrieval_time_success})

# utils_retr.make_scatter_plot(axes[2,0], {"rho of bias and tau": rho_bias_tau}, \
#     {"average time of retrieval (success)": average_retrieval_time_success})

# utils_retr.make_scatter_plot(axes[2,1], {"rho of postive weight and tau": rho_pos_weight_tau}, \
#     {"average time of retrieval (success)": average_retrieval_time_success})

# utils_retr.make_scatter_plot(axes[2,2], {"rho of negative weight and tau": rho_neg_weight_tau}, \
#     {"average time of retrieval (success)": average_retrieval_time_success})

# utils_retr.make_scatter_plot(axes[2,1], {"bias": bias}, \
#     {"pattern_degree": pattern_deg})

utils_retr.make_scatter_plot(axes[2,0], {"bias": bias}, \
    {"absolute weight": mean_weight}, {"tau": t_sets[-6,:]}, color_bin_n=7)

utils_retr.make_scatter_plot(axes[1,1], \
    {"absolute weight": mean_weight}, {"tau": t_sets[-6,:]})

utils_retr.make_scatter_plot(axes[1,2], \
    {"bias": bias}, {"tau": t_sets[-6,:]})


# utils_retr.make_scatter_plot(axes[2,1], {"positive weight": mean_weight_pos}, \
#     {"negative weight": mean_weight_neg})

# utils_retr.make_scatter_plot(axes[1,0], {"tau 0": tau_0}, {"tau_optimal_1": tau_opt_1})
# utils_retr.make_scatter_plot(axes[1,1], {"tau 0": tau_0}, {"tau_optimal_2": tau_opt_2})
#utils_retr.make_scatter_plot(axes[1,2], {"tau 0": tau_0}, {"tau_optimal_3": tau_opt_3})

# check if all retreive states' neuron are near 0 and 1?
# axes[1,0].hist(retrieved_patterns.flatten())
plt.savefig("retrieval plot of n tau set.jpg")

plt.show(block=False)


# calculate the ratio of 'violated' weight of certain stored pattern positive and negative separately:
alpha_plus_array = np.zeros(n_pattern)
alpha_plus_check_array = np.zeros(n_pattern)
alpha_minus_array = np.zeros(n_pattern)
for ii in range(n_pattern):
    P = patterns[ii]
    violation_matrix = (2*P-1)[:,None]*weight*(2*P-1)[None,:]
    act_neuron = P>0.5
    inact_neuron = P<0.5
    alpha_plus_array = np.sum(violation_matrix[act_neuron,:]>1e-7)/(n_neuron-1)/(act_neuron.sum())
    alpha_plus_check_array = np.sum(violation_matrix[act_neuron,:]<-1e-7)/(n_neuron-1)/(act_neuron.sum())
    alpha_minus_array = np.sum(violation_matrix[inact_neuron,:]>1e-7)/(n_neuron-1)/(inact_neuron.sum())

alpha_plus = alpha_plus_array.mean()
alpha_plus_check = alpha_plus_check_array.mean()
alpha_minus = alpha_minus_array.mean()

##############################################

print("--- %s seconds ---" % (time.time() - start_time))

print("the biggest different in success retrieval rate = {}".format(per_success_retrieval.max()-per_success_retrieval.min()))

print("end")


    

