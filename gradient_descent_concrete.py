from importlib.metadata import requires
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import softmax
import matplotlib.animation as animation
torch.set_default_tensor_type(torch.DoubleTensor)

class model_arrival(nn.Module):
    def __init__(self, s, e, balanced_factor = 0.1, T=96, total_vehicles=100, battery_capacity=8, power_capacity=8, u = 0.6, decay_rate = 0.06):
        super().__init__()
        self.s = s
        self.e = e
        self.T = T
        self.N = total_vehicles
        self.B = battery_capacity
        self.P = power_capacity
        self.u = u
        self.decay_rate = decay_rate
        self.balanced_factor = balanced_factor
        self.y = nn.Parameter(torch.ones(self.e-self.s+1))
        self.softmax = nn.Softmax(dim=0)
    
    def detect(self, y):
        s, e, T = self.s, self.e, self.T
        y = y.detach().numpy()
        required_energy = (0.9-0.2) * self.B
        t_end = [-1 for _ in range(e-s+1)]
        for idx in range(e-s+1):
            for t in range(1,T-s+1):
                p = self.P * np.ones(t)
                if t <= e-s+1:
                    n = np.cumsum(y[:t]*self.N)
                else:
                    n = np.concatenate((np.cumsum(y*self.N), self.N * np.ones(t-len(y))),axis=0)
                for idx2 in range(idx):
                    depart_t = t_end[idx2]
                    n[depart_t:] -= y[idx2] * self.N
                u_available = np.minimum(np.ones(t), p/n)[idx:]
                if sum(u_available) >= required_energy:
                    t_end[idx] = t
                    break
            if t_end[idx] == -1: break
        return t_end
            
    def evaluate(self):
        s, e, T = self.s, self.e, self.T
        y = self.softmax(self.y)
        depart = self.detect(y)
        y = y * self.N
        n = torch.cat((torch.cumsum(y, dim=0), self.N*torch.ones(T-e)), 0)
        energy_loss = torch.tensor([0.], requires_grad=True)
        for i in range(e-s+1):
            if depart[i] == -1:
                if i > 0:
                    energy_loss = torch.sum(torch.stack([torch.sum(y[ii]/n[ii:depart[ii]]*torch.minimum(n[ii:depart[ii]]*self.u, self.P*torch.ones(depart[ii]-ii))) - 0.7*self.B*y[ii] for ii in range(i)]))
                energy = torch.sum(torch.minimum(n*self.u, self.P*torch.ones(T-s+1))) - energy_loss
                t1 = (self.T - self.s) * torch.ones(self.e-self.s+1) - torch.cumsum(torch.ones(self.e-self.s+1), dim=0)
                t2 = torch.tensor(depart[:i])
                stay = torch.sum(y[i:] * t1[i:]/self.N) + torch.sum(y[:i]*t2/self.N) 
                # record s
                self.energy = np.round(energy.detach().numpy(),2)
                self.stay = np.round(stay.detach().numpy(),2)
                self.n_plot = np.concatenate((np.zeros(s), n.detach().numpy()),axis=0)
                self.y_plot = np.concatenate((np.zeros(s), y.detach().numpy(), np.zeros(T-e)),axis=0)
                return  - (energy - self.balanced_factor * stay)
            else:
                t = depart[i]
                n[t:] -= y[i]
    

if __name__ == "__main__":
    solver = model_arrival(30, 50, balanced_factor=5)
    optimizer = optim.Adam(solver.parameters(), lr=1e-3)
    i = 0
    loss = []
    figure, axes = plt.subplots(1,2, figsize=(8,4))
    while i < 10000:
        i = i + 1
        L = solver.evaluate()
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        if i > 5000:
            optimizer.param_groups[0]["lr"] = 1e-5
        if i % 10 == 0:
            loss.append(L.detach().numpy())
            anno1 = plt.annotate('balanced_factor = 0.1', xy=(-1, 0.9), xycoords='axes fraction',color='black')
            anno = plt.annotate(f'step:{i}, \nenergy:{solver.energy}\nstay:{solver.stay}\nloss:{loss[-1]}', xy=(0.8, 0.9), xycoords='axes fraction',color='black')
            axes[0].plot(solver.y_plot,color='gray')
            #print(solver.n_plot)
            axes[0].plot(solver.n_plot, color='orange')
            axes[0].set_ylim([-10,120])
            axes[0].set_xlim([-1, 100])
            axes[1].plot(loss, color='blue')
            plt.pause(0.0001)
            anno.remove()
            axes[0].clear()
            
    anno = plt.annotate('step:%d'%i, xy=(0.85, 0.9), xycoords='axes fraction',color='black')
    plt.pause(0)


