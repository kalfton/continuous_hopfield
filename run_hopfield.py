import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from hopfield_class_torch import Hopfield_network
import torch
import pickle
import time
import utils_pytorchV2 as utils

random.seed(0)
torch.manual_seed(0)
n_neuron = 50
n_pattern = 40

start_time = time.time()

# with open('trained_network_torch_back_prop.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network1, patterns) = data_saved

patterns = utils.make_pattern(n_pattern, n_neuron)
network1 = Hopfield_network(n_neuron, dt=0.01)

# network1, success, patterns = utils.train_equalibium_prop(network1, patterns, lr=0.01, max_loop_train=100, dt_train = 0.5, gamma=50)
network1, success, patterns_stored = utils.train_back_prop(network1, patterns, lr=0.01, n_step = 5, dt_train = 0.5)
# network1, success, patterns = utils.train_PLA(network1, patterns, lr=0.01, k1 = 0.0, k2 = 2)

#Evaluation: 
# sanity check:
pattern_id=21
with torch.no_grad():
    final_pattern, converge, retrieval_time= network1.evolve_batch(patterns[[pattern_id]])
    print(torch.abs(final_pattern-patterns[pattern_id]))

plt.figure()
plt.hist(torch.abs(final_pattern-patterns[pattern_id]))
plt.show(block=False)

# save the trained weight bias and stored patterns
# data = [network1, patterns]
# with open('trained_network_torch.pickle', 'wb') as f:
#     pickle.dump(data, f)

print("--- %s seconds ---" % (time.time() - start_time))




#binary search to find max capacity:
n_repeat = 3
start_n = 0
end_n = int(2*n_neuron)
n_pattern_set = []
success_ratio = []

flag_n = -1

while (start_n <= end_n):
    mid_n = int(start_n+np.floor((end_n-start_n)/2))
    print(f"evaluating at n_patter = {mid_n}")

    success_count = 0
    all_count = 0
    for j in range(n_repeat):
        patterns = utils.make_pattern(mid_n, n_neuron)
        network = Hopfield_network(n_neuron, dt=0.01)
        network1, success = utils.train_equalibium_prop(network, patterns, lr=0.01, dt_train = 0.5)
        #network1, success = utils.train_back_prop(network, patterns, lr=0.01, n_step = 2, dt_train = 0.5)
        #network1, success = utils.train_PLA(network, patterns, lr=0.01, k1 = 0.0, k2 = 2)

        all_count += 1
        if success:
            success_count+=1
            break
    success_ratio.append(success_count/all_count)
    n_pattern_set.append(mid_n)

    if success_ratio[-1]<=0:
        end_n = mid_n - 1
    else:
        flag_n = mid_n
        start_n = mid_n + 1

plt.figure()
plt.scatter(n_pattern_set, success_ratio)
plt.xlabel('n pattern')
plt.ylabel('success rate')
plt.title(f'capacity: {flag_n/n_neuron}')
plt.show(block=False)
plt.savefig('capacity_analysis_PLA.png')

with open('logging.txt','a') as f:
    f.write("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))
print('end')


        







