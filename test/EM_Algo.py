import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.linalg import solve_discrete_lyapunov
# from scipy.signal import correlate, hilbert


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("default", category=RuntimeWarning)

def rot(w): 
    return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]], dtype=np.float64)

def likelihood(x, xm, Pinv, dN, mu_c, Ts, beta):
    '''
    Likelihood function:

    log p({x_{k}} | H(k)) = 
                    -(1/2) (x_{k} - \alpha R(w) x_{k-1|k-1} )^T P_{k|k-1}^{-1} (x_{k} - \alpha R(w) x_{k-1|k-1} ) 
                    + \sum_{c=1}^{C} ( dN^c(k) (\mu_c + x_{1,k}) - exp(\mu_c + x_{1,k})*\Delta)
    '''
    dot2 = (-1/2)*np.linalg.multi_dot([(x - xm).T, Pinv, x - xm])
    L = dN*(mu_c + beta*x[0] + np.log(Ts)) - np.exp(mu_c + beta*x[0])*Ts + dot2
    return -L

def jacobian(x, xm, Pinv, dN, mu_c, Ts, beta):
    '''
    
    Jacobian = -P_{k|k-1}^{-1} (x_{k} - x_{k|k-1} ) + [1 0]* (dN^c(k) - exp(\mu_c) + [\betabeta 0]x_{k|k-1}\Delta)*\beta
    
    '''
    jac = -np.dot(Pinv, x - xm)
    jac[0] += (dN - np.exp(mu_c + beta*x[0])*Ts)*beta
    return  -jac

def hessian(x, xm, Pinv, dN, mu_c, Ts, beta):
    '''
    Hessian = -P_{k|k-1}^{-1}  - [[1 0], [0 0]]* (exp(\mu_c) + [beta 0]x_{k|k-1}*\Delta) * \beta^2
    '''
    
    hess = -Pinv.copy()
    hess[0,0] -= np.exp(mu_c + beta*x[0])*Ts*np.power(beta, 2)
    return -hess

def max_L(x0, xp_k, Pinv_k, y, mu_c, Delta, beta, method='trust-krylov'):
    return minimize(likelihood, x0, args=(xp_k, Pinv_k, y, mu_c, Delta, beta), 
                           jac=jacobian, hess=hessian, method = method)

def KF_step(x_m_k1, P_m_k1, alpha, sigma, R, mu, Delta, beta, y_k, method='trust-krylov'):
    '''
    Single step of Kalman Filtering
    '''
    #Prediction Step
    xp_k = alpha * np.dot(R, x_m_k1)
    Pp_k = np.power(alpha, 2) * np.linalg.multi_dot([R, P_m_k1, R.T]) + sigma*np.eye(2)
    
    Pinv_k = np.linalg.pinv(Pp_k)

    # Measurement step : maximizing likelihood function
    # Using x_{k|k-1} as initial estimate for minimize function
    res = max_L(x_m_k1, xp_k, Pinv_k, y_k, mu, Delta, beta, method=method)
    
    if not res.success:
        print("Estimation of $x_{m, k}$  failed to converge : " + res.message)
        raise AssertionError("Estimation of $x_{m, k}$  failed to converge : " + res.message)

    Pm_k = np.linalg.pinv(res.hess) # Negative not required because hessian function minimizes the "negative likelihood".
    return xp_k, res.x, Pp_k, Pm_k

def backward_pass(alpha, R, x_p, x_m, P_p, P_m, K):
    '''
    Backward pass smoother

    returns smoothed estimates and backward gains.
    '''
    AG = np.zeros((2,2,K))
    x_b = np.zeros((2,K))
    P_b = np.zeros((2,2,K))
    x_b[:, -1] = x_m[:, -1].copy()
    P_b[:, :, -1] = P_m[:, :, -1].copy()
    for k in range(K-1, 0, -1):
        A_k1 = np.linalg.multi_dot([P_m[:, :, k-1], alpha*R.T, np.linalg.pinv(P_p[:, :, k])])
        AG[:, :, k-1] = A_k1

        x_b[:, k-1] = x_m[:, k-1] + np.dot(A_k1, x_b[:, k] - x_p[:, k])
        P_b[:, :, k-1] = P_m[:, :, k-1] + np.linalg.multi_dot([A_k1, P_b[:, :, k] - P_p[:, :, k], A_k1.T])
        
    return x_b, P_b, AG

def KFilter(x0, y, mu, beta, sigma, alpha, F, Delta, K=None):
    '''
    Full filter. Executes forward pass and backward pass and returns prediction, measurement and backward-smoothed estimates
    '''
    if not K:
        K = y.shape[1]
    
    omega = 2*np.pi*F/1000
    R = rot(omega)
    
    x_p = np.zeros((2,K))
    x_m = np.zeros((2,K))
    
    P_p = np.zeros((2,2,K))
    P_m = np.zeros((2,2,K))

    P_m[:, :, 0] = (sigma/(1 - np.power(alpha, 2)))*np.eye(2) #steady state calculation of P_m as an estimate of P_0|0
    x_m[:, 0] = x0.copy()

    #Forward Pass : Kalman Filter
    for k in range(1, K):
        xp_k, xm_k, Pp_k, Pm_k = KF_step(x_m[:, k-1], P_m[:, :, k-1], 
                                                 alpha, sigma, R, mu, Delta, beta, y[k], method='trust-krylov')
        x_p[:, k], P_p[:, :, k], x_m[:, k], P_m[:, :, k] = xp_k, Pp_k, xm_k, Pm_k

    #Backward Pass
    x_b, P_b, AG = backward_pass(alpha, R, x_p, x_m, P_p, P_m, K)
    
    return x_p, x_m, x_b, P_p, P_m, P_b, AG


def E_step(x_b, P_b, AG, K):
    '''
    Takes in smoothed estimates and gains to output A,B,C used to estimate parameters
    '''
    A = np.zeros((2,2))
    for k in range(1, K):
        A += P_b[:, :, k-1] + np.dot(x_b[:, k-1], x_b[:, k-1].T)
        
    B = np.zeros((2,2))
    for k in range(1, K):
        B += np.dot(P_b[:, :, k], AG[:, :, k-1].T) + np.dot(x_b[:, k], x_b[:, k-1].T)
        
    C = np.zeros((2,2))
    for k in range(1, K):
        C += P_b[:, :, k] + np.dot(x_b[:, k], x_b[:, k].T)
        
    return A, B, C

def M_step(A, B, C, x_b, P_b, dN, K, beta, Delta):

    '''
    Updates parameters given smoothed estimates
    '''
    omega_h = np.arctan2(B[1,0] - B[0,1], np.trace(B))
    alpha_h = (np.trace(B)*np.cos(omega_h) + (B[1,0] - B[0,1])*np.sin(omega_h)) / np.trace(A)
    sigma_h = (np.trace(C) - np.power(alpha_h, 2)*np.trace(A))/(2*K)
    
    exp_sum = np.exp(beta*x_b[0, :] + np.power(beta,2)*P_b[0,0, : ]/2).sum()
    mu_h = np.log(np.sum(dN)) - np.log(exp_sum) - np.log(Delta)

    return mu_h, alpha_h, omega_h, sigma_h


def run_EM(y, x0, mu0, beta, sigma0, alpha0, F0, Delta, iters = 100, method='trust-krylov'):

    K = len(y)

    
    omega0 = 2*np.pi*F0*Delta
    
    params = pd.DataFrame(columns=['$\\mu$', '$\\alpha$', '$\\omega$', '$\\sigma^2$'])
    params.loc[0, :] = [mu0, alpha0, omega0, sigma0]

    error_log = pd.DataFrame(columns = ['n_iter', 'x0', 'mu_h', 'alpha_h', 'omega_h', 'sigma_h', 'msg'])
    
    E_steps = {'x_p': [],'x_m': [],'x_b': [],'P_p': [],'P_m': [],'P_b': [], 'BG':[], 'A':[], 'B':[], 'C':[]}
    
    for itr in range(1, iters+1):
        print(itr, end=' ')

        try:
            # Filtering to get x and P estimates
            x_pred, x_meas, x_smooth, P_pred, P_meas, P_smooth, BG = KFilter(x0, y, mu0, beta, sigma0, alpha0, F0, Delta, K)
            
            for k,v in zip(['x_p','x_m','x_b','P_p','P_m','P_b', 'BG'], 
                           [x_pred, x_meas, x_smooth, P_pred, P_meas, P_smooth, BG]):
                E_steps[k].append(v)
            
            # Maximizing to update param values
            # E-step
            A,B,C = E_step(x_smooth, P_smooth, BG, K)
            for k,v in zip(['A', 'B', 'C'], [A, B, C]):
                E_steps[k].append(v)
    
            # M-step
            mu_h, alpha_h, omega_h, sigma_h = M_step(A, B, C, x_smooth, P_smooth, y, K, beta, Delta)
            params.loc[itr, :] = [mu_h, alpha_h, omega_h, sigma_h]
    
    
            #Update for next iter
            F0 = omega_h/(2*np.pi*Delta) # since filter takes in F and not omega
            mu0, alpha0, omega0, sigma0 = mu_h, alpha_h, omega_h, sigma_h
        except AssertionError as err:
            error_log.loc[error_log.shape[0], :] = [itr, x0, mu0, alpha0, omega0, sigma0, err.__str__()]

    return params, E_steps, [mu_h, alpha_h, F0, omega_h, sigma_h], error_log

def get_params(N = 1000, random = True, plot = False, print_params = False):
    '''
    Generate parameters randomly or returns fixed parameters.
    Also plots and displays for convenience.
    '''
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
        Delta = 1/Fs
        
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
        rng = np.random.default_rng(42)
        mu = -4.5
        beta = 10
        sigma = 0.001
        alpha = 0.973
        F = 60
        Fs = 1000
        Delta = 1/Fs
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
        for name, var in zip(['N', 'Fs', 'Delta', 'alpha', 'F', 'omega', 'sigma', 'mu', 'beta', 'R'], 
                             [N, Fs, Delta, alpha, F, omega, sigma, mu, beta, R]):
            print(f'{name} : {np.round(var, 4)}')

    return X, y, N, Fs, Delta, alpha, F, omega, R, sigma, mu, beta


if __name__ == "__main__":
    N = 2000
    X, y, N, Fs, Delta, alpha, F, omega, R, sigma, mu, beta = get_params(N=N, random = False, print_params=True)

    params, E_steps, [mu_est, alpha_est, F_est, omega_est, sigma_est], error_log = run_EM(y, x0, 
                                                                           mu, beta, sigma, alpha, F, Delta, 
                                                                           iters = 10, method='trust-krylov')
    

    
