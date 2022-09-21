import numpy as np
import matplotlib.pyplot as plt
from util import SimpleGA, CMAES, PEPG, OpenES
import cma
import random
from scipy.special import softmax
import collections
import plotly.graph_objects as go


def charging_schedule(y, balanced_fac=0.5, method='ES', verbose=False):
    N = 100
    B = 6
    maxu = 0.5
    decay_rate = 0.06
    T = 48
    P = 10
    balanced_factor = balanced_fac

    stay = 0
    y = softmax(y)
    n = y * N
    x_init, final_energy = 0.2*B*n, 0.9*B
    x_step = np.copy(x_init)
    #print(n)
    for k in range(T):
        u = np.zeros(T)
        V = [i for i in range(k+1) if x_step[i] <= final_energy*n[i] - 1e-3]
        totalev = np.sum([n[i] for i in V])
        stay += totalev
        if method == 'fix': budget = P
        for i in V:
            if method == 'ES':
                u[i] = min(min(P/totalev*n[i], n[i]*(maxu-decay_rate*x_step/n[i])), final_energy*n[i] - x_step[i])
            elif method == 'fix':
                u[i] = min(n[i]*maxu, final_energy*n[i] - x_step[i])
                if budget < u[i]: 
                    u[i] = budget
                    break
                budget -= u[i]
        x_step = x_step + u
    energy = np.sum(x_step-x_init)
    if verbose:
        print(energy, stay)
        return energy, stay
    return -(energy - balanced_factor*stay/N)


def search(balanced_fac, method, solution=False):
    x0 = [0]*48 # initial solution
    sigma0 = 1    # initial standard deviation to sample new solutions
    x, es = cma.fmin2(charging_schedule, x0, sigma0, args=(balanced_fac, method, False))
    energy, stay = charging_schedule(x, balanced_fac, method, True)
    if solution:
        return x
    print(energy, stay)
    return energy, stay

if __name__ == '__main__':
    balanced_fac = 1
    method = 'ES'
    search(balanced_fac, method)