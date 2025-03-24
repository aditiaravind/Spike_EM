import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import correlate, hilbert

def get_params(N = 1000, random = True, plot = False, print_params = False):

    # Randomly choose input constants that fit expected criteria for observation signal
    if random:
        rng = np.random.default_rng()
        mu = 2*rng.random() - 5  # range (-5, -3)
        beta = 8 # beta is modified below to ensure that y spikes approx comes from peaks of obs; beta range (1, 21)
        alpha = 0.248*rng.random() + 0.75 # range (0.75, 0.999)
        F = 100*rng.random() + 20 # range (20, 120)

        # Keeping these fixed for now
        sigma = 0.001 
        Fs = 1000
        
        omega = 2*np.pi*F/Fs
        R = rot(omega)
        
        # Inputs 
        u = rng.normal(0, scale=np.sqrt(sigma), size=(2,N))
        X = np.zeros_like(u)
        X[:, 0] = u[:, 0]
        for k in range(1,N):
            X[:, k] = alpha*np.dot(R, X[:, k-1]) + u[:, k]
        obs = np.exp(mu + beta*X[0])

        # Ensuring obs is in the *right* range of ~ (0, 1) so np.exp(mu + beta*x[0]) isn't too large
        while obs.max() < 0.75:
            if beta > 20:
                break
            beta += 1
            obs = np.exp(mu + beta*X[0])
    
        while obs.max() >= 1.05:
            if beta < 2:
                break
            beta -= 1
            obs = np.exp(mu + beta*X[0])
        
        # y spikes whenever obs > random_number
        y = np.zeros_like(obs)
        for i, yi in enumerate(obs):
            if yi > rng.random():
                y[i] = 1
    
    #fixed parameter values
    else:
        rng = np.random.default_rng(4)
        mu = -4.5
        beta = 10
        sigma = 0.001
        alpha = 0.973
        F = 8.5
        Fs = 1000
        omega = 2*np.pi*F/Fs
        R = rot(omega)

        u = rng.normal(0, scale=np.sqrt(sigma), size=(2,N))
        X = np.zeros_like(u)
        X[:, 0] = u[:, 0]
        for k in range(1,N):
            X[:, k] = alpha*np.dot(R, X[:, k-1]) + u[:, k]
        obs = np.exp(mu + beta*X[0])
        y = np.zeros_like(obs)
        for i, yi in enumerate(obs):
            if yi > rng.random():
                y[i] = 1
        
    if plot:
        fig, ax = plt.subplots()
        ax.vlines(np.where(y)[0], ymin=0, ymax = 1, ls='--', color='grey', lw=0.85)
        ax.plot(obs)
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('$e^{\\mu + \\beta x_{1}}$')
        ax.set_title(f'Firing rate : {y.mean()*Fs} Hz', fontsize=12)

    if print_params:
        print(f'X dims : {X.shape}')
        print(f'y dims : {y.shape}')
        for name, var in zip(['N', 'Fs', 'alpha', 'F', 'omega', 'sigma', 'mu', 'beta', 'R'], 
                             [N, Fs, alpha, F, omega, sigma, mu, beta, R]):
            print(f'{name} : {np.round(var, 4)}')

    return X, y, N, Fs, alpha, F, omega, R, sigma, mu, beta




def check_gradient(f, grad, x0, args):

    delta = np.random.random(size=x0.shape)
    out = pd.DataFrame(columns = ['$||\\delta||$', '$||\\delta||/||x_0||$', 'abs_e', 'rel_e', 'close'])
    for mag in range(10):
        lhs = f(x0+delta, *args) - f(x0, *args)
        rhs = np.vdot(delta, grad(x0, *args))
        abs_e = np.abs(lhs - rhs)
        rel_e = abs_e / (np.abs(lhs)) if lhs != 0 else np.nan
        out.loc[mag+1] = [np.linalg.norm(delta), np.linalg.norm(delta)/np.linalg.norm(x0), 
                          abs_e, rel_e, np.allclose(rhs, lhs)]
        delta /= 10
        # x0 = np.random.random(size=x0.shape)
    return out