import torch
import torch.optim as optim
import numpy as np
from torch import nn
from utils_pytorch import Hopfield_network_torch
from utils import make_pattern
from numpy import random
from matplotlib import pyplot as plt


max_loop = 50000
min_error = 1e-5
n_neuron = 200
n_pattern = 25
torch.manual_seed(0)
random.seed(0)


network2 = Hopfield_network_torch(n_neuron, dt = 0.1)
patterns = torch.from_numpy(make_pattern(n_pattern, n_neuron)).type(torch.float)

print(network2)

#BPTT Training: 
optimizer = optim.SGD(network2.parameters(), lr=0.01)
lossfun = nn.MSELoss()


training_loss = []
n_loop = 0
while n_loop<max_loop:
    losses = torch.zeros(n_pattern)
    for i in range(n_pattern):
        P = patterns[i]
        optimizer.zero_grad()
        retrieve_P = network2(P)
        losses[i] = lossfun(P, retrieve_P)
    loss = torch.mean(losses)
    if loss<min_error:
        break
    loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(network2.parameters(), 100)

    if torch.any(torch.isnan(network2.h2h.weight.grad)):
        print('pause')

    optimizer.step()

    if n_loop%100==0:
        print(f'Step: {n_loop}, Loss: {loss}')
        training_loss.append(loss.detach().numpy().item())

    #For debugging: 
    if torch.isnan(network2.h2h.weight.min()):
        print('pause')

    n_loop+=1

# sanity check:
with torch.no_grad():
    final_pattern, converge= network2.evolve(patterns[4])
    print(torch.abs(final_pattern-patterns[4]))

plt.figure()
plt.hist(torch.abs(final_pattern-patterns[4]))
plt.show(block=False)

plt.figure()
plt.plot(np.arange(0, max_loop, 100), training_loss)
plt.show(block = False)

print('end')