import torch
import torch.nn as nn
import torch.random as random

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

class Hopfield_network_torch(nn.Module):
    # Thought: In training and evaulation, we use different dt?
    # Reasoning: During training dt can be large make the training process easier.
    # but during evaluation (memory retrieval) the dynamics should be more precise.
    def __init__(self, n_neuron, n_step = 1, dt = None):
        super().__init__()
        self.n_neuron = n_neuron
        self.tau = torch.ones(n_neuron)
        self.beta = torch.ones(n_neuron)*100
        if dt is None:
            alpha = torch.ones(n_neuron)
        else:
            alpha = dt/self.tau

        self.alpha = alpha
        self.oneminusalpha = 1-alpha
        self.h2h = nn.Linear(n_neuron, n_neuron)
        self.n_step = n_step
        # self.state_x = torch.zeros(n_neuron)
        # self.state_g = self.g(self.state_x, self.beta)

        # Try to initiate the weight and bias of the network for reproducibility
        #torch.nn.init.uniform_(self.h2h.weight)

    # def init_state(self, state_g):
    #     self.state_g = state_g
    #     self.state_x = self.g_inverse(state_g,self.beta)
    
    def W(self):
        return self.h2h.weight.detach().numpy()

    def b(self):
        return self.h2h.bias.detach().numpy()

    def g(self, x, beta): #activation_func
        return torch.sigmoid(beta*x)
        #return 1/(1+torch.exp(-beta*x))

    def g_inverse(self, g, beta): # inverse of g
        return (1/beta) * torch.log(g/(1-g))
    
    def recurrence(self, state_x, state_g):
        pre_activation = self.h2h(state_g)
        state_x = self.alpha*pre_activation + self.oneminusalpha*state_x
        state_g = self.g(state_x, self.beta)
        return state_x, state_g

    def forward(self, state_g):
        #self.init_state(state_g)
        state_x = self.g_inverse(state_g,self.beta)       
        for _ in range(self.n_step):
            state_x, state_g = self.recurrence(state_x, state_g)

        return state_g
        



