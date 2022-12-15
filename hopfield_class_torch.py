import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

class Symmetric_Linear(nn.Module):
    def __init__(self, n_neuron):
        super().__init__()
        self.n_neuron = n_neuron
        W = torch.normal(torch.zeros(n_neuron, n_neuron), 0.01)
        W = W+W.T
        W = W-torch.diag(torch.diag(W))
        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(torch.normal(torch.zeros(n_neuron), 0.01))

    def forward(self, x):
        return F.linear(x, (self.weight+self.weight.T)/2-torch.diag(torch.diag(self.weight)), self.bias)
    
    def extra_repr(self) -> str:
        return 'features={}, bias={}'.format(
            self.n_neuron, self.bias is not None
        )


class Hopfield_network(nn.Module):
    max_loop = 20000
    min_error = 1e-7

    def __init__(self, n_neuron, dt=0.1):
        super().__init__()
        self.n_neuron = n_neuron
        self.tau = nn.Parameter(torch.ones(n_neuron), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(n_neuron)*10,requires_grad=False)
        # if dt_train is None:
        #     alpha = torch.ones(n_neuron)
        # else:
        #     alpha = dt_train/self.tau
        # self.alpha = alpha
        # self.oneminusalpha = 1-alpha
        self.h2h = Symmetric_Linear(n_neuron)
        self.max_x = 50/self.beta.max().item()
        self.dt = dt

        self.state_x = nn.Parameter(torch.zeros(n_neuron),requires_grad=False)
        self.state_g = nn.Parameter(self.g(self.state_x, self.beta),requires_grad=False)

    def W(self):
        return self.h2h.weight.detach()

    def b(self):
        return self.h2h.bias.detach()

    def set_W(self, W:torch.tensor):
        with torch.no_grad():
            self.h2h.weight = nn.parameter.Parameter(W)

    def set_b(self, b:torch.tensor):
        with torch.no_grad():
            self.h2h.bias = nn.Parameter(b)
    #TODO check how many parameters are there?

    def g(self, x, beta): #activation_func
        return torch.sigmoid(beta*x)

    def g_inverse(self, g, beta): # inverse of g
        return (1/beta) * torch.log(g/(1-g))

    def Lagrangian(self, x, beta):
        torch.sum((1/beta)*torch.log(1+torch.exp(beta*x)))


    def energy_func(self, state_x, state_g):
        return torch.sum((state_x-self.b)*state_g) - \
            self.Lagrangian(state_x, self.b) - 0.5*state_g@self.W()@state_g
    

    def init_states(self, state_g):
        self.state_g = state_g
        self.state_x = self.g_inverse(state_g, self.beta)
        self.state_x = torch.clip(self.state_x, -self.max_x, self.max_x)


    def set_tau(self, tau):
        self.tau = tau

    def forward(self, state_g, n_step, dt_train):  # This is the forward function for training
        state_x = self.g_inverse(state_g,self.beta)       
        for _ in range(n_step):
            state_x, state_g = self.recurrence(state_x, state_g, dt_train)
        return state_g


    def recurrence(self, state_x, state_g, dt):
        pre_activation = self.h2h(state_g)
        state_x = (dt/self.tau)*pre_activation + (1-dt/self.tau)*state_x
        state_x = torch.clip(state_x, -self.max_x, self.max_x)
        state_g = self.g(state_x, self.beta)
        return state_x, state_g

    def recurrence_F(self, state_x, state_g, target, gamma, dt):
        # dx = dt/self.tau*(self.W()@self.state_g+self.b()-self.state_x-gamma*(self.state_g-target))
        # self.state_x = self.state_x+dx
        # self.state_g = self.g(self.state_x, self.beta)

        pre_activation = self.h2h(state_g)-gamma*(self.state_g-target)
        state_x = (dt/self.tau)*pre_activation + (1-dt/self.tau)*state_x
        state_x = torch.clip(state_x, -self.max_x, self.max_x)
        state_g = self.g(state_x, self.beta)
        return state_x, state_g

    def evolve(self, state_g, target=None, gamma=None):
        with torch.no_grad():
            self.init_states(state_g)
            n_loop=1
            while n_loop<=self.max_loop:
                n_loop = n_loop+1
                old_state_x = self.state_x.clone()
                if target==None:
                    self.state_x, self.state_g = self.recurrence(self.state_x, self.state_g, self.dt)
                else:
                    self.state_x, self.state_g = self.recurrence_F(self.state_x, self.state_g, target, gamma, self.dt)

                if torch.max(torch.abs(self.state_x-old_state_x))<self.min_error:
                    break
            
            if n_loop>=self.max_loop:
                converge = False
            else:
                converge = True
        return self.state_g.detach(), converge



                
