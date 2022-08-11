from importlib.metadata import requires
from tkinter import Y
import torch
from torch import nn
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


class model_arrival(nn.Module):

    def __init__(self, s, e, T=96, total_vehicles=100, battery_capacity=8, power_capacity=8, u = 0.6, decay_rate = 0.06):
        super().__init__()
        self.s = s
        self.e = e
        self.T = T
        self.N = total_vehicles
        self.B = battery_capacity
        self.P = power_capacity
        self.u = u
        self.decay_rate = decay_rate
        self.y = nn.Parameter(torch.ones(e-s+1))
        self.softmax = nn.Softmax(dim=0)
    
    def evaluate(self, y):
        arrival = y
        depart = self.T * torch.ones(self.N)
        initial_state = 0.2 * self.B * torch.ones((self.N))
        final_energy =  0.9 * self.B * torch.ones((self.N))
        required_energy = final_energy - initial_state
        return self.allocate_charging(arrival, depart, initial_state, final_energy)
    
    def allocate_charging(self, arrival_time, depart_time, initial_state, final_energy):
        initial_state_EDF = torch.clone(initial_state)
        u_mat = torch.tensor(0., requires_grad=True)

        for t in range(int(arrival_time[0]), self.T-1):
            power_budget = self.P #Change this for variable case
            vehicle_ending_index = (arrival_time < t).sum()
            step_initial_SOC = initial_state_EDF[:vehicle_ending_index]
            final_energy_needed = final_energy[:vehicle_ending_index]
            depart_schedule = depart_time[:vehicle_ending_index]
            u_val = torch.zeros_like(step_initial_SOC)

            num_active_sessions = 0
            for i in range(vehicle_ending_index):
                if depart_schedule[i] >= t:
                    if step_initial_SOC[i] <= final_energy[i]-1e-3:
                        num_active_sessions += 1

            if num_active_sessions==0:
                num_active_sessions=1
            shared_power = self.P/num_active_sessions

            for i in range(vehicle_ending_index):
                if depart_schedule[i] >= t:
                    if step_initial_SOC[i] <= final_energy[i]:
                        u_val[i]=shared_power
            
            updated_val = torch.minimum(u_val, torch.ones_like(u_val) * self.u - self.decay_rate * step_initial_SOC)
            updated_val = torch.minimum(updated_val, final_energy_needed-step_initial_SOC) 
            initial_state_EDF[:vehicle_ending_index] += updated_val
            u_mat += torch.sum(updated_val)
        return -u_mat
    
    def allocate_arrival(self):
        s, e, N = self.s, self.e, self.N
        x0 = self.softmax(self.y)
        sorted, indices = torch.sort(x0,descending=True)
        print(x0)
        arrival = torch.sum(torch.stack([(s+i)*torch.ones((x0[i]*N).int())  for i in range(e-s+1) ]))
        print(arrival, arrival.requires_grad)
        # idx = 0
        # while idx < N:
        #     for i in indices:
        #         arrival[idx:idx+1] = s+i
        #         idx += 1
        #         if idx >= N:
        #             break
        return arrival


if __name__ == '__main__':
    solver = model_arrival(s=40,e=60)
    a = solver.allocate_arrival()
    a = a.sum()
    a.backward()
    print(solver.y.grad)
    # i = 0
    # while True:
    #     i = i + 1
    #     y.requires_grad_()
    #     L = solver.evaluate(y)
    #     L.backward()
    #     y = y.detach()
    #     if i % 5 == 0:
    #         print(y)

