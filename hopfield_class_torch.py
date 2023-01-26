import numpy as np
from numpy import random
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import scipy

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
    max_loop = 200000
    min_error = 1e-7
    min_error_g = 1e-9

    def __init__(self, n_neuron, dt=0.1):
        super().__init__()
        self.n_neuron = n_neuron
        self.tau = nn.Parameter(torch.ones(n_neuron),requires_grad=False)
        self.beta = nn.Parameter(torch.ones(n_neuron)*10,requires_grad=False)

        self.h2h = Symmetric_Linear(n_neuron)
        self.max_x = 50/self.beta.max().item()
        self.dt = dt

        # self.state_x = torch.zeros(n_neuron)
        # self.state_g = self.g(self.state_x, self.beta)

    def W(self):
        return self.h2h.weight.detach()

    def b(self):
        return self.h2h.bias.detach()

    def tau_np(self):
        return self.tau.detach().numpy()

    def set_W(self, W:torch.tensor):
        with torch.no_grad():
            self.h2h.weight = nn.parameter.Parameter(W)

    def set_b(self, b:torch.tensor):
        with torch.no_grad():
            self.h2h.bias = nn.Parameter(b)

    def set_tau(self, tau):
        with torch.no_grad():
            if type(tau)==np.ndarray:
                self.tau = nn.Parameter(torch.from_numpy(tau).float(),requires_grad=False)
            elif type(tau)==torch.Tensor:
                self.tau = nn.Parameter(tau.detach(),requires_grad=False)
            else:
                raise TypeError("Wrong input type")

    def g(self, x, beta): #activation_func
        return torch.sigmoid(beta*x)

    def g_inverse(self, g, beta): # inverse of g
        return (1/beta) * torch.log(g/(1-g))

    def g_derivative(self, x, beta):
        g = self.g(x, beta)
        return beta*g*(1-g)

    def Lagrangian(self, x, beta):
        return torch.sum((1/beta)*torch.log(1+torch.exp(beta*x)))


    def energy_func(self, state_x, state_g, target=None, gamma=None):
        assert state_x.shape == state_g.shape, "state_x and state_g should have the same dimension"
        assert state_x.dim() ==1, "Currently the energy function only accept 1 dim input"
            
        if target==None:
            return torch.sum((state_x-self.b())*state_g) - \
                self.Lagrangian(state_x, self.b()) - 0.5*state_g@self.W()@state_g.T
        else:
            return torch.sum((state_x-self.b())*state_g) - \
                self.Lagrangian(state_x, self.b()) - 0.5*state_g@self.W()@state_g.T +\
                gamma*torch.sum((state_g-target)**2)
    

    # def init_states(self, state_g):
    #     self.state_g = state_g
    #     self.state_x = self.g_inverse(state_g, self.beta)
    #     self.state_x = torch.clip(self.state_x, -self.max_x, self.max_x)


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
        pre_activation = self.h2h(state_g)-gamma*(state_g-target)
        state_x = (dt/self.tau)*pre_activation + (1-dt/self.tau)*state_x
        state_x = torch.clip(state_x, -self.max_x, self.max_x)
        state_g = self.g(state_x, self.beta)
        return state_x, state_g

    # def evolve(self, state_g, target=None, gamma=None):
    #     with torch.no_grad():
    #         self.init_states(state_g)
    #         n_loop=0
    #         while n_loop<self.max_loop:
    #             n_loop = n_loop+1
    #             old_state_x = self.state_x.clone()
    #             if target==None:
    #                 self.state_x, self.state_g = self.recurrence(self.state_x, self.state_g, self.dt)
    #             else:
    #                 self.state_x, self.state_g = self.recurrence_F(self.state_x, self.state_g, target, gamma, self.dt)

    #             if torch.max(torch.abs(self.state_x-old_state_x))<self.min_error:
    #                 break
            
    #         if n_loop>=self.max_loop:
    #             converge = False
    #         else:
    #             converge = True
    #     return self.state_g.detach(), converge, n_loop*self.dt


    def evolve_batch(self, state_g:torch.Tensor, target=None, gamma=None):
        #state_g is a tensor with size n_patterns *  n_neurons
        assert state_g.dim()==2, "state_g must be a two dim tensor"

        n_pattern = state_g.shape[0]
        state_x = self.g_inverse(state_g,self.beta)
        with torch.no_grad():
            n_loop=0
            converge_matrix = torch.zeros(n_pattern, self.max_loop)
            while n_loop<self.max_loop:
                old_state_x = state_x
                old_state_g = state_g
                if target==None:
                    state_x, state_g = self.recurrence(state_x, state_g, self.dt)
                else:
                    state_x, state_g = self.recurrence_F(state_x, state_g, target, gamma, self.dt)

                converge_matrix[:,n_loop] = torch.max(torch.abs(state_x-old_state_x), dim = 1)[0]

                n_loop = n_loop+1
                if torch.max(torch.abs(state_x-old_state_x))<self.min_error:
                    break

            n_step = torch.sum(converge_matrix>self.min_error, dim=1)+1 # min(max(n_step),self.max_loop))==n_loop
            converge = n_step<self.max_loop
        return state_g.detach(), converge, n_step*self.dt
    
    # #Testing new alg. here
    # def recurrence_normalized(self, state_x, state_g, dt, use_tau = False):
    #     dynamics = self.g_derivative(state_x, self.beta)*(self.h2h(state_g)-state_x)
    #     if use_tau:
    #         vec_norm = torch.linalg.norm(dynamics, 2, dim = -1, keepdim=True)
    #         dynamics_weighted = (1/self.tau) * dynamics
    #         unconstrain_norm = torch.linalg.norm(dynamics_weighted, 2, dim = -1, keepdim=True)
    #         dynamics = dynamics_weighted/unconstrain_norm*vec_norm
        
    #     B = torch.linalg.norm(dynamics, 2, dim = -1, keepdim=True)
    #     dynamics = dynamics/B
    #     state_x = state_x + dt*dynamics
    #     state_x = torch.clip(state_x, -self.max_x, self.max_x)
    #     state_g = self.g(state_x, self.beta)
    #     return state_x, state_g

    # def evolve_batch_normalized(self, state_g:torch.Tensor, use_tau = False):
    #     #state_g is a tensor with size n_patterns *  n_neurons
    #     assert state_g.dim()==2, "state_g must be a two dim tensor"

    #     n_pattern = state_g.shape[0]
    #     state_x = self.g_inverse(state_g,self.beta)
    #     with torch.no_grad():
    #         n_loop=0
    #         converge_matrix = torch.zeros(n_pattern, self.max_loop)
    #         while n_loop<self.max_loop:
    #             old_state_g = state_g

    #             state_x, state_g = self.recurrence_normalized(state_x, state_g, self.dt, use_tau)
    #             # state_x, state_g = self.recurrence(state_x, state_g, self.dt)

    #             converge_matrix[:,n_loop] = torch.max(torch.abs(state_g-old_state_g), dim = 1)[0]

    #             n_loop = n_loop+1
    #             if torch.max(torch.abs(state_g-old_state_g))<self.min_error_g:
    #                 break

    #         n_step = torch.sum(converge_matrix>self.min_error_g, dim=1)+1 # min(max(n_step),self.max_loop))==n_loop
    #         converge = n_step<self.max_loop
    #     return state_g.detach(), converge, n_step*self.dt



    





                
