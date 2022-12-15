import numpy as np
from numpy import random
from matplotlib import pyplot as plt

class Hopfield_network:
    max_loop = 10000
    min_error = 1e-8

    def __init__(self, n_neuron):
        self.n_neuron = n_neuron
        self.W = np.zeros([n_neuron, n_neuron])
        self.b = np.zeros(n_neuron) # threshold for each neuron
        self.tau = np.ones(n_neuron)
        self.beta = np.ones(n_neuron)*10
        self.max_x = 50/self.beta.max()
        self.state_x = np.zeros(n_neuron)
        self.state_g = self.g(self.state_x, self.beta)

    def g(self, x, beta): #activation_func
        return 1/(1+np.exp(-beta*x))

    def g_inverse(self, g, beta): # inverse of g
        return (1/beta) * np.log(g/(1-g))

    def Lagrangian(self, x, beta):
        np.sum((1/beta)*np.log(1+np.exp(beta*x)))


    def energy_func(self):
        return np.sum((self.state_x-self.b)*self.state_g) - \
            self.Lagrangian(self.state_x, self.b) - 0.5*self.state_g@self.W@self.state_g
    

    def init_state_g(self, state_g):
        self.state_g = state_g
        self.state_x = self.g_inverse(state_g, self.beta)

    def set_tau(self, tau):
        self.tau = tau
        
    def forward(self, dt=0.01):
        # Kaining's note : Use Runge–Kutta method for higher accuracy.
        # Euler method: 
        dx = dt/self.tau*(self.W@self.state_g+self.b-self.state_x)
        self.state_x = self.state_x+dx
        # clip to prevent state_x goes to inf
        self.state_x = np.clip(self.state_x, -self.max_x, self.max_x)
        self.state_g = self.g(self.state_x, self.beta)
        

    def evolve(self, P):
        self.init_state_g(P)
        n_loop=1
        while n_loop<=self.max_loop:
            n_loop = n_loop+1
            old_state_g = self.state_g.copy()
            self.forward()

            if np.max(np.abs(self.state_g-old_state_g))<self.min_error:
                break

            # for debugging:
            if np.any(np.isnan(self.state_g)):
                print('pause')
        
        if n_loop>=self.max_loop:
            converge = False
        else:
            converge = True
        return self.state_g.copy(), converge

    def forward_F(self, target, gamma, dt=0.01):
        dx = dt/self.tau*(self.W@self.state_g+self.b-self.state_x-gamma*(self.state_g-target))
        self.state_x = self.state_x+dx
        self.state_g = self.g(self.state_x, self.beta)

    def evolve_F(self, P, target, gamma):
        self.init_state_g(P)
        n_loop=1
        while n_loop<=self.max_loop:
            n_loop = n_loop+1
            old_state_g = self.state_g.copy()
            self.forward_F(target, gamma)

            if np.max(np.abs(self.state_g-old_state_g))<self.min_error:
                break
        
        if n_loop>=self.max_loop:
            converge = False
        else:
            converge = True
        return self.state_g.copy(), converge



                
