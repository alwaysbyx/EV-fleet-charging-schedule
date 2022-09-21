from ast import iter_fields
from termios import TAB1
import numpy as np
import random
import collections
from scipy.special import softmax
import matplotlib.pyplot as plt

TOU_price = np.zeros(48)
TOU_price[:12]=0.51
TOU_price[12:]=0.43

class cross_entropy_time:
    def __init__(self, e, balanced_factor=1,  Nsamples=1000, rho=0.05,maxite=100, total_vehicles=100, P=4, B=6, maxu=2, decay_rate=0, seed=0, method='ES'):
        np.random.seed(seed)
        random.seed(seed)
        self.mu = np.random.rand(e)
        self.sigma = np.eye(e)
        self.e = e
        self.rho = rho
        self.Nsamples= Nsamples
        self.maxite = maxite
        self.N = total_vehicles
        self.require = None
        self.P = P
        self.B = B
        self.x_init = np.zeros(e)
        self.decay_rate = decay_rate
        self.maxu = maxu
        self.method = method
        self.balanced_factor = balanced_factor

    def search(self, show_animation=True):
        if show_animation:
            figure, axes = plt.subplots(1,2, figsize=(8,4))
        scores_record = []
        for ite in range(self.maxite):
            samples = self.get_samples()
            #print('get samples',samples)
            samples, scores, total = self.choose_samples(samples)
            #print('score samples',samples,scores)
            self.update_parameters(samples)
            scores_record.append(np.mean(scores))
            self.score = np.mean(scores)
            print(f'iteration={ite}, score={np.mean(scores)}, total={np.mean(total)}')
            if show_animation:
                y = softmax(samples[0])*self.N
                energy, stay, _ = self.get_function_value(samples[0], True)
                anno1 = plt.annotate(f'balanced_factor = {self.balanced_factor}', xy=(-1, 1.03), xycoords='axes fraction',color='black')
                anno = plt.annotate(f'step:{ite}, \nenergy:{energy}\nstay:{stay}\nloss:{scores_record[-1]}', xy=(0.8, 0.9), xycoords='axes fraction',color='black')
                axes[0].plot(y, color='orange')
                #print(sum(y))
                axes[0].set_ylim([-1, 10])
                axes[0].set_xlim([-1, 50])
                axes[1].plot(scores_record, color='blue')
                plt.pause(0.0001)
                anno.remove()
                axes[0].clear()

    def get_sample(self):
        '''generate one sample according to parameters'''
        return np.diag(np.random.normal(self.mu, self.sigma))

    def get_samples(self):
        '''generate n samples according to parameters'''
        return [self.get_sample() for _ in range(self.Nsamples)]

    def choose_samples(self, samples):
        '''choose the elite set from n samples generated'''
        num = int(self.rho * self.N)
        scores = []
        for sample in samples:
            scores.append(self.get_function_value(sample))
        scoredsamples = list(zip(scores,list(samples)))
        scoredsamples = sorted(scoredsamples, key=lambda x: -x[0])
        samples = [sample for score,sample in scoredsamples]
        scores = [score for score,sample in scoredsamples]
        self.best = samples[0]
        return samples[:num+1], scores[:num+1], scores
    
    def update_parameters(self,samples):
        '''using the new elite set to update parameters'''
        mu = np.mean(np.array(samples),axis=0)
        sigma = np.std(np.array(samples),axis=0)
        #print('mu',mu, 'sigma',sigma)
        self.mu = mu
        self.sigma = np.eye(self.e)
        np.fill_diagonal(self.sigma, sigma)

    def get_function_value(self, y, verbose = False, process=False):
        method = self.method
        maxu = self.maxu
        decay_rate = self.decay_rate
        T = self.e
        P = self.P
        stay, charging_cost = 0, 0
        if not process:
            y = softmax(y)
        n = y * self.N
        x_init, final_energy = self.x_init * n, 0.9*self.B
        x_step = np.copy(x_init)
        u_record = [0 for _ in range(T)]
        stay_record = [0 for _ in range(T)]
        for k in range(T):
            u = np.zeros(T)
            V = [i for i in range(k+1) if x_step[i] <= final_energy*n[i] - 1e-3]
            stay_record[k] = sum([n[i] for i in V])
            totalev = np.sum([n[i] for i in V])
            stay += totalev
            if method == 'fix': budget = P
            for i in V:
                if method == 'ES':
                    u[i] = min(min(P/totalev*n[i], n[i]*(maxu-decay_rate*x_step[i]/n[i])), final_energy*n[i] - x_step[i])
                elif method == 'fix':
                    u[i] = min(n[i]*(maxu-decay_rate*x_step[i]/n[i]), final_energy*n[i] - x_step[i])
                    if budget < u[i]: 
                        u[i] = budget
                        break
                    budget -= u[i]
            x_step = x_step + u
            charging_cost = charging_cost + np.sum(u)*TOU_price[k]
            u_record[k] = sum(u)
        energy = np.sum(x_step-x_init)
        self.energy = energy
        self.stay = stay
        self.stay_record = stay_record
        self.u_record = u_record
        if verbose:
            print(energy, stay, charging_cost)
            return self
        #return - max(min(self.P*self.e, 420)-energy, 0) - self.balanced_factor*stay/self.N
        return  energy - self.balanced_factor*stay/self.N
        #return - max(420-energy, 0) - self.balanced_factor * (max(u_record)-min(u_record))/np.average(u_record)#- self.balanced_factor*stay/self.N

if __name__ == '__main__':
    solver = cross_entropy_time(48, balanced_factor=10,  Nsamples=200, rho=0.05, maxite=100, total_vehicles=100, P=10, B=6, maxu=0.5, method='fix')
    solver.search()