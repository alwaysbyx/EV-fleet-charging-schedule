from distutils.util import change_root
from importlib.metadata import requires
from pickletools import uint2
from tkinter import N, Y
from cross_entropy_sampling import TOU_price
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.set_default_tensor_type(torch.DoubleTensor)

TOU_price = np.zeros(48)
TOU_price[:12]=0.5
TOU_price[12:]=0.3

class model_arrival(nn.Module):
    def __init__(self, e, strategy="ES", balanced_factor=0, T=30, decay_rate=0, total_vehicles=100, battery_capacity=8, power_capacity=8, u = 0.6, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.T = T
        self.e = e
        self.balanced_factor = balanced_factor
        self.N = total_vehicles
        self.B = battery_capacity
        self.P = power_capacity
        self.u = u
        #self.y = nn.Parameter(20*torch.rand(e))
        self.y = nn.Parameter(20*torch.ones(e))
        self.softmax = nn.Softmax(dim=0)
        
        #self.x = torch.clip(0.5*torch.rand(e), 0.1, 0.5)*self.B
        self.final_energy = 0.9*self.B
        self.decay = decay_rate

        self.strategy = strategy
    
    def forward(self):
        y = self.softmax(self.y)
        # with torch.no_grad():
        #     torch.clip(self.y, 1e-3, 1e3)
        # y = self.y / torch.sum(self.y)
        cost = 0
        for i in range(100):
            x = torch.FloatTensor(self.T).uniform_(0.1,0.5)*self.B
            cost += self.allocate_charging(y,x)
        return cost / 100

    def allocate_charging_withsoc(self, y, x):
        n = y*self.N
        x_step = x
        x_init = x.clone()
        n2 = torch.cumsum(n, dim=0)
        stay = 0
        final_energy = torch.tensor(self.final_energy)
        u = torch.tensor(self.u)
        for k in range(self.T):
            for i in range(self.N):
                if i <= n2[k] and x_step[i] < final_energy:
                    u = torch.minimum(u, final_energy-x_step)
                    x_step += u
                    stay += 1
        return torch.sum(x_step-x_init)-stay

    
    def allocate_charging(self, y, x):
        n = y * self.N
        x_step = x.clone()*n
        x_init = x.clone()*n
        t_indices = torch.cumsum(torch.ones(self.e), dim=0)-1
        stay = 0
        charging_cost = 0
        for k in range(self.T):
            V = torch.where((x_step < self.final_energy*n) & (t_indices <= k), 1, 0)
            if self.strategy == 'ES':  # equal sharing
                u = torch.minimum(torch.minimum(self.P / torch.sum(V*n) * n, n*(self.u-self.decay*x_step/n)), self.final_energy*n - x_step) * V
            elif self.strategy == 'fix':
                u = torch.minimum(n*(self.u-self.decay*x_step/n), self.final_energy*n - x_step) * V
                #u2 = torch.minimum(n*self.u, self.final_energy*n - x_step) * V
                if torch.sum(u) >= self.P+0.1:
                    for idx in range(1, self.T+1):
                        if torch.sum(u[:idx]) >= self.P:
                            u[idx:] -= u[idx:]
                            u[idx-1] = self.P-torch.sum(u[:idx-1])
                            break
                try:
                    assert torch.sum(u) <= self.P + 0.1
                except Exception as e:
                    print(torch.sum(u), u)
                    return
            x_step += u
            stay += torch.sum(n*V)
            charging_cost += torch.sum(u)*TOU_price[k]
        energy = torch.sum(x_step - x_init)
        self.n_plot = n.detach().numpy()
        self.y_plot = y.detach().numpy()
        self.energy = energy.detach().numpy()
        self.stay = stay.detach().numpy()
        self.charging_cost = charging_cost.detach().numpy()
        #return -(energy - self.balanced_factor * torch.maximum(stay-500, torch.tensor(0)))
        #return max(min(self.P*self.e, 420)-energy, 0) + self.balanced_factor * stay / self.N
        return -energy + self.balanced_factor * stay / self.N


def train(solver, balanced_factor, show_animation=False):
    optimizer = optim.Adam(solver.parameters(), lr=5e-2)
    loss = []
    if show_animation:
        figure, axes = plt.subplots(1,2, figsize=(8,4))
    for i in tqdm(range(500)):
        L = solver()
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        loss.append(L.detach().numpy())
        if i == 0:
            print(solver.energy, solver.stay)
        if i == 15:
            optimizer = optim.Adam(solver.parameters(), lr=5e-3)
        if show_animation:
            anno1 = plt.annotate(f'balanced_factor = {balanced_factor}', xy=(-1, 1.03), xycoords='axes fraction',color='black')
            anno = plt.annotate(f'step:{i}, \nenergy:{solver.energy}\nstay:{solver.stay}\nloss:{loss[-1]}\ncost:{solver.charging_cost}', xy=(0.8, 0.9), xycoords='axes fraction',color='black')
            axes[0].plot(solver.y_plot,color='gray')
            #print(solver.n_plot)
            axes[0].plot(solver.n_plot, color='orange')
            axes[0].set_ylim([-1,100])
            axes[0].set_xlim([-1, 35])
            axes[1].plot(loss, color='blue')
            plt.pause(0.0001)
            anno.remove()
            axes[0].clear()
    if show_animation:
        anno = plt.annotate('step:%d'%i, xy=(0.85, 0.9), xycoords='axes fraction',color='black')
        plt.pause(0)
    solver.loss = loss
    return solver


if __name__ == '__main__':
    balanced_factor = 20
    solver = model_arrival(strategy='fix', seed=40, e=48, u=0.5, decay_rate=0.0, total_vehicles=100, balanced_factor=balanced_factor, battery_capacity=6, power_capacity=10, T=48)
    train(solver=solver, balanced_factor=balanced_factor, show_animation=True)