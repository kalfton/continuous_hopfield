import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from hopfield_class_torch import Hopfield_network
import torch
import torch.nn as nn
import pickle
import time
import utils_pytorchV2 as utils
import warnings

random.seed(10)
torch.manual_seed(10)
n_neuron = 50
n_pattern = 30

start_time = time.time()

# compare the weight

# with open('data/trained_network_torch_eq_prop.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network1, patterns) = data_saved

# with open('data/trained_network_torch_back_prop.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network2, patterns) = data_saved

# with open('data/trained_network_torch_PLA.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network3, patterns) = data_saved

patterns = utils.make_pattern(n_pattern, n_neuron)
network1 = Hopfield_network(n_neuron, dt=0.01)
network2 = Hopfield_network(n_neuron, dt=0.01)
network3 = Hopfield_network(n_neuron, dt=0.01)

network1, success, stored_patterns1 = utils.train_equalibium_prop(network1, patterns, lr=0.01, max_loop_train=100, dt_train = 0.5, gamma=50)
if not success:
    warnings.warn("Not successfully train the network")
network2, success, stored_patterns2 = utils.train_back_prop(network2, patterns, lr=0.01, n_step = 2, dt_train = 0.5)
if not success:
    warnings.warn("Not successfully train the network")
network3, success, stored_patterns3 = utils.train_PLA(network3, patterns, lr=0.01, k1 = 0.0, k2 = 2)
if not success:
    warnings.warn("Not successfully train the network")



# Compare the difference in parameters of the three trained network:
with torch.no_grad():
    W1=network1.W()
    W2=network2.W()
    W3=network3.W()
    b1=network1.b()
    b2=network2.b()
    b3=network3.b()
    W1 = torch.concat([W1,b1[None,:]],dim=0)
    W2 = torch.concat([W2,b2[None,:]],dim=0)
    W3 = torch.concat([W3,b3[None,:]],dim=0)
    # normalization:
    W1 = (W1)/torch.std(W1)
    W2 = (W2)/torch.std(W2)
    W3 = (W3)/torch.std(W3)

    #plot
    z_min=-4
    z_max=4
    fig, axes = plt.subplots(3,3, figsize = [19,10])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
    axes[0,0].matshow(W1,cmap='bwr', vmin=z_min, vmax=z_max)
    axes[0,0].set_title("equilibium prop")

    axes[0,1].matshow(W2,cmap='bwr',vmin=z_min, vmax=z_max)
    axes[0,1].set_title("back prop")

    c=axes[0,2].matshow(W3,cmap='bwr',vmin=z_min, vmax=z_max)
    axes[0,2].set_title("PLA")
    colorbar = plt.colorbar(c, ax=axes[0,2])

    # calculate the difference between each pair of weights
    lossfun = nn.MSELoss()
    dist = lossfun(W1, W2)
    axes[1,0].matshow(W1-W2,cmap='bwr',vmin=z_min, vmax=z_max)
    axes[1,0].set_title(f"distance = {dist:.6f}")

    dist = lossfun(W2, W3)
    axes[1,1].matshow(W2-W3,cmap='bwr',vmin=z_min, vmax=z_max)
    axes[1,1].set_title(f"distance = {dist:.6f}")

    dist = lossfun(W3, W1)
    c = axes[1,2].matshow(W3-W1,cmap='bwr',vmin=z_min, vmax=z_max)
    axes[1,2].set_title(f"distance = {dist:.6f}")
    colorbar = plt.colorbar(c, ax=axes[1,2])

    
    plt.show(block=False)
    plt.savefig('weight_comparison.jpg')


# save the trained weight bias and stored patterns
data = [network1, stored_patterns1]
with open('data/trained_network_torch_eq_prop.pickle', 'wb') as f:
    pickle.dump(data, f)

data = [network2, stored_patterns2]
with open('data/trained_network_torch_back_prop.pickle', 'wb') as f:
    pickle.dump(data, f)

data = [network3, stored_patterns3]
with open('data/trained_network_torch_PLA.pickle', 'wb') as f:
    pickle.dump(data, f)

print("--- %s seconds ---" % (time.time() - start_time))