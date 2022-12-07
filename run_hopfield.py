import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from hopfield_class import Hopfield_network
import pickle
import time
import utils

n_neuron = 100
n_pattern = 10


start_time = time.time()

# with open('trained_network_PLA.pickle', 'rb') as f:
#     data_saved = pickle.load(f)
# (network1, patterns) = data_saved

patterns = utils.make_pattern(n_pattern, n_neuron)
network1 = Hopfield_network(n_neuron)

network1, converge = utils.train_equalibium_prop(network1, patterns)
#network1, converge = utils.train_PLA(network1, patterns)

# sanity check:
final_pattern, _ = network1.evolve(patterns[4])
print(np.abs(final_pattern-patterns[4]))

# save the trained weight bias and stored patterns
data = [network1, patterns]
with open('trained_network.pickle', 'wb') as f:
    pickle.dump(data, f)

print("--- %s seconds ---" % (time.time() - start_time))
print('end')






