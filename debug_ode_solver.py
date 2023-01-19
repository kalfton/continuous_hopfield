import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define vectors, indices and parameters
resetV = -0.1
nIN = 3
incIN = nIN
ylen = nIN*(incIN)
indIN = np.arange(0,ylen,incIN)
INs = np.arange(0,nIN)    
gI = -0.4    
Ileak = 0.5


# Define heaviside function for synaptic gates (just a continuous step function)
def heaviside(v,thresh):
    H =  0.5*(1 +np.tanh((v-thresh)/1e-8))
    return H

# Define event functions and set them as terminal
def event(t, y,z):
    return y[indIN[0]] - 2
event.terminal = True

# def event2(t,y):
#     return y[indIN[1]] - 2
# event2.terminal = True

# def event3(t,y):
#     return y[indIN[2]] - 2
# event3.terminal = True

#ODE function
def Network(t,y,z):

    V1 = y[0]
    n11 = y[1]
    n12 = y[2]
    V2 = y[3]
    n21 = y[4]
    n22 = y[5]
    V3 = y[6]
    n31 = y[7]
    n32 = y[8]

    H = heaviside(np.array([V1,V2,V3]),INthresh)

    dydt = [V1*V1 - gI*n11*(V2)- gI*n12*(V3)+0.5,
            H[1]*5*(1-n11) - (0.9*n11),
            H[2]*5*(1-n12) - (0.9*n12),
            V2*V2 -gI*n21*(V1)- gI*n22*(V3),
            H[0]*5*(1-n21) - (0.9*n21),
            H[2]*5*(1-n22) - (0.9*n22),
            V3*V3 -gI*n31*(V1)- gI*n32*(V2),
            H[0]*5*(1-n31) - (0.9*n31),
            H[1]*5*(1-n32) - (0.9*n32)
            ]

    return dydt

# Preallocation of some vectors (mostly not crucial)
INthresh = 0.5
dydt = [0]*ylen
INheavies = np.zeros((nIN,))
preInhVs = np.zeros((nIN,))          
y = np.zeros((ylen,))

allt = []
ally = []
t = 0
end = 100

# Integrate until an event is hit, reset the spikes, and use the last time step and y-value to continue integration



net = solve_ivp(Network, (t, end), y, args = (0,), events= [event])
allt.append(net.t)
ally.append(net.y)
if net.status == 1:

    t = net.t[-1]
    y = net.y[:, -1].copy()
    for i in INs:    
        if net.t_events[i].size != 0:
            y[indIN[i]] = resetV
            print('reseting V%d' %(i+1))



# Putting things together and plotting
Tp = np.concatenate(allt)
Yp = np.concatenate(ally, axis=1)
fig = plt.figure(facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(Tp, Yp[0].T)
ax2.plot(Tp, Yp[3].T)
ax3.plot(Tp, Yp[6].T)
plt.subplots_adjust(hspace=0.8)

plt.show()