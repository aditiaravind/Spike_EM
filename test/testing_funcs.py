import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import correlate, hilbert



def split_err(err):
    mad = re.findall(r"Max absolute difference: ([\d.eE+-]+)", err.__str__())[0]
    mrd = re.findall(r"Max relative difference: ([\d.eE+-]+)", err.__str__())[0]
    msg = err.__str__().split('\n')[2]
    tol_msg = err.__str__().split('\n')[1]
    return msg, tol_msg, mrd, mad


def test_max_L(n_runs = 100):

    log = pd.DataFrame(columns = ['run', 'success', 'y', 'mu', 'beta', 'alpha', 'x0', 'x1', 'xm0', 'xm1', 'x1_check', 'x2_check', 'x3_check','P00', 'P01', 'P10', 'P11', 'H00', 'H01', 'H10', 'H11', 'msg', 'tol_msg', 'mrd', 'mad'])
    for n_run in range(n_runs):
        try:
            rng = np.random.default_rng()
            x = rng.random(2)
            mu = 20*rng.random() - 10
            beta = rng.random()*8
            # beta = 1
            Delta = 1e-3
            alpha = 0.5*rng.random() + 0.5
            
            # P = gen_test_P()
            P = (0.001/(1 - np.power(alpha, 2)))*np.eye(2)

            # P_try = 
            Pinv = np.linalg.pinv(P)
            obs = np.exp(mu + beta*x[0])
            if obs > rng.random():
                y = 1
            else:
                y = 0
        
            res = max_L(x, Pinv, y, mu, Delta, beta)
            xm = res.x
            hess = res.hess
            
            hess_inv = np.linalg.pinv(hess)
        
            #Hand checking
            modifier = beta*(y - np.exp(mu + beta*xm[0])*Delta)
            x1_check = x[0] + P[0,0]*modifier
            x2_check = x[1] + P[1,0]*modifier
            x3_check = x[1] + P[0,1]*modifier
        
            J = -Pinv.copy()
            J[0,0] -= np.power(beta, 2)*np.exp(mu + beta*xm[0])*Delta
        
            P_check = -np.linalg.pinv(J)
            P_check2 = -J
            
            
            np.testing.assert_allclose(x2_check, xm[1],  err_msg = 'Checking xm[1], P[1,0]')
            np.testing.assert_allclose(x3_check, xm[1],  err_msg = 'Checking xm[1], P[0,1]')
            np.testing.assert_allclose(P_check2, hess,  err_msg = 'Checking hess == -J')
            np.testing.assert_allclose(P_check, hess_inv,  err_msg = 'Checking hess == -J^(-1)')
            np.testing.assert_allclose(x1_check, xm[0],  err_msg = 'Checking xm[0]')
            log.loc[n_run, :] = [n_run, True, y, mu, beta, alpha, x[0], x[1], xm[0], xm[1], 
                                 x1_check, x2_check, x3_check, P[0,0], P[0,1], P[1,0], P[1,1], 
                                 hess_inv[0,0], hess_inv[0,1], hess_inv[1,0], hess_inv[1,1], *[np.nan]*4]
        except AssertionError as e:
            log.loc[n_run, :] = [n_run, False, y, mu, beta, alpha, x[0], x[1], xm[0], xm[1], 
                                 x1_check, x2_check, x3_check, P[0,0], P[0,1], P[1,0], P[1,1], 
                                 hess_inv[0,0], hess_inv[0,1], hess_inv[1,0], hess_inv[1,1], *split_err(e)]

    
    return log

def test_KF(N=1000, K=1000, plot=True):

    rng = np.random.default_rng(24)

    #Input constants
    # mu = 2*rng.random() - 5 #20*rng.random() - 10
    # beta = 8 # beta = rng.random()*8
    # sigma = 0.001
    # Delta = 1e-3
    # alpha = 0.248*rng.random() + 0.75
    # F = 100*rng.random() + 20
    # omega = 2*np.pi*F/1000
    # R = rot(omega)

    mu = -4.5 #20*rng.random() - 10
    beta = 8 # beta = rng.random()*8
    sigma = 0.001
    Delta = 1e-3
    alpha = 0.973
    F = 60
    omega = 2*np.pi*F/1000
    R = rot(omega)
    
    # Inputs 
    u = rng.normal(0, scale=np.sqrt(sigma), size=(2,N))
    X = np.zeros_like(u)
    
    X[:, 0] = u[:, 0]
    for k in range(1,N):
        X[:, k] = alpha*np.dot(R, X[:, k-1]) + u[:, k]

    obs = np.exp(mu + beta*X[0])
    while obs.max() < 0.75:
        if beta > 20:
            break
        beta += 1
        obs = np.exp(mu + beta*X[0])

    while obs.max() >= 1.2:
        if beta < 2:
            break
        beta -= 1
        obs = np.exp(mu + beta*X[0])
        
    y = np.zeros_like(obs)
    for i, yi in enumerate(obs):
        if yi > rng.random():
            y[i] = 1
    
    print(f'X dim : {X.shape}')
    print(f'y dim : {y.shape}')
    for var, name in zip([mu, alpha, F, omega, sigma, beta], ['mu', 'alpha', 'F', 'omega', 'sigma^2', 'beta']):
        print(f'{name} = {var:.4f}')

    if plot:
        plt.vlines(np.where(y)[0], ymin=0, ymax=1, color='grey', ls='--', lw=0.9)
        plt.plot(np.exp(mu + beta*X[0]))
        plt.title(f'Firing rate : {y.mean()/Delta :.2f} Hz')
        plt.show()
        
    x_p = np.zeros((2,K))
    x_m = np.zeros((2,K))
    
    P_p = np.zeros((2,2,K))
    P_m = np.zeros((2,2,K))

    P_m[:, :, 0] = (sigma/(1 - np.power(alpha, 2)))*np.eye(2) #gen_test_P()
    x_m[:, 0] = X[:, 0].copy()

    log = pd.DataFrame(columns = ['success', 'y', 'xdiff', 'xp0', 'xp1', 'xm0', 'xm1', 'x1_check', 'x2_check', 
                                              'H00', 'H11', 'msg', 'tol_msg', 'mrd', 'mad'])
    for k in range(1, K):
        xp_k, xm_k, Pp_k, Pm_k = KF_step(x_m[:, k-1], P_m[:, :, k-1], alpha, sigma, R, mu, Delta, beta, y[k], method='trust-krylov')

        x_p[:, k], P_p[:, :, k], x_m[:, k], P_m[:, :, k] = xp_k, Pp_k, xm_k, Pm_k
        #Hand calc
        modifier = beta*(y[k] - np.exp(mu + beta*xm_k[0])*Delta)
        x1_check = xp_k[0] + Pp_k[0,0]*modifier
        x2_check = xp_k[1] + Pp_k[1,0]*modifier
    
        J = -np.linalg.pinv(Pp_k)
        J[0,0] -= np.power(beta, 2)*np.exp(mu + beta*xm_k[0])*Delta
        P_check = -np.linalg.pinv(J)

        try:
            np.testing.assert_allclose(Pm_k, P_check,  err_msg = 'Checking Pm_k')
            np.testing.assert_allclose(xm_k[1], x2_check,  err_msg = 'Checking xm_k[0]')
            np.testing.assert_allclose(xm_k[0], x1_check,  err_msg = 'Checking xm_k[0]')
            
            log.loc[k, :] = [True, y[k], np.abs(xm_k[0] - x1_check), xp_k[0], xp_k[1], xm_k[0], xm_k[1], 
                                                                 x1_check, x2_check, Pm_k[0,0], Pm_k[1,1], *[np.nan]*4]
        except AssertionError as e:
            log.loc[k, :] = [False, y[k], np.abs(xm_k[0] - x1_check), xp_k[0], xp_k[1], xm_k[0], xm_k[1], 
                                                                 x1_check, x2_check, Pm_k[0,0], Pm_k[1,1], *split_err(e)]
    
    return log, X, x_p, x_m, P_p, P_m, y, [mu, alpha, F, omega, sigma, beta]
        
def test_jac(n_runs = 100):

    log = pd.DataFrame(columns = ['run', 'success', 'y', 'mu', 'alpha', 'beta', 'x0', 'x1', 'xm0', 'xm1', 
                                  'P00', 'P11', 'H00', 'H11', 'HInv00', 'HInv11', 'msg', 'tol_msg', 'mrd', 'mad'])
    for n_run in range(n_runs):
        try:
            rng = np.random.default_rng()
            x = rng.random(2)
            mu = 2*rng.random() - 5
            beta = rng.random()*8
            # beta = 1
            Delta = 1e-3
            alpha = 0.238*rng.random() + 0.75
            
            # P = gen_test_P()
            P = (0.001/(1 - np.power(alpha, 2)))*np.eye(2)

            # P_try = 
            Pinv = np.linalg.pinv(P)
            obs = np.exp(mu + beta*x[0])
            if obs > rng.random():
                y = 1
            else:
                y = 0
                
            res = minimize(likelihood, x, args=(x, Pinv, y, mu, Delta, beta), 
                           jac=jacobian, hess=hessian, method = 'trust-krylov')
            
            est = minimize(likelihood, x, args=(x, Pinv, y, mu, Delta, beta), 
                           jac=jacobian, hess='2-point', method = 'trust-krylov')
            
            np.testing.assert_allclose(res.fun, est.fun,  err_msg = 'Comparing L outputs, res.fun, est.fun')
            np.testing.assert_allclose(hessian(res.x, x, Pinv, y, mu, Delta, beta), res.hess,  
                                       err_msg = 'Comparing res.hess with function')
            np.testing.assert_allclose(jacobian(res.x, x, Pinv, y, mu, Delta, beta), res.jac,  
                                       err_msg = 'Comparing res.jac with function')
            np.testing.assert_allclose(approx_fprime(est.x, jacobian, 1e-6, x, Pinv, y, mu, Delta, beta), res.hess,
                                       err_msg = 'Comparing res.hess with fprime(jacobian)')

            hess_inv = np.linalg.pinv(res.hess)
            
            log.loc[n_run, :] = [n_run, True, y, mu, alpha, beta, x[0], x[1], res.x[0], res.x[1], 
                                 P[0,0], P[1,1], res.hess[0,0], res.hess[1,1], hess_inv[0,0], hess_inv[1,1], *[np.nan]*4]
        except AssertionError as e:
            log.loc[n_run, :] = [n_run, False, y, mu, alpha, beta, x[0], x[1], res.x[0], res.x[1], 
                                 P[0,0], P[1,1], res.hess[0,0], res.hess[1,1], hess_inv[0,0], hess_inv[1,1], *split_err(e)]
    
    return log
    
def test_param(N=1000):
    '''
    Make beta match whatever random mu is used
    '''
    rng = np.random.default_rng()

    #Input constants
    mu = 2*rng.random() - 5 #20*rng.random() - 10
    beta = 8 # beta = rng.random()*8
    sigma = 0.001
    Delta = 1e-3
    alpha = 0.248*rng.random() + 0.75
    F = 100*rng.random() + 20
    omega = 2*np.pi*F/1000
    R = rot(omega)
    
    # Inputs 
    u = rng.normal(0, scale=np.sqrt(sigma), size=(2,N))
    X = np.zeros_like(u)
    
    X[:, 0] = u[:, 0]
    for k in range(1,N):
        X[:, k] = alpha*np.dot(R, X[:, k-1]) + u[:, k]

    obs = np.exp(mu + beta*X[0])

    while obs.max() < 0.75:
        if beta > 20:
            break
        beta += 1
        obs = np.exp(mu + beta*X[0])

    while obs.max() >= 1.2:
        if beta < 2:
            break
        beta -= 1
        obs = np.exp(mu + beta*X[0])
    
    y = np.zeros_like(obs)
    for i, yi in enumerate(obs):
        if yi > rng.random():
            y[i] = 1

    
    print(f'X dim : {X.shape}')
    print(f'y dim : {y.shape}')
    for var, name in zip([mu, alpha, F, omega, sigma, beta], ['mu', 'alpha', 'F', 'omega', 'sigma^2', 'beta']):
        print(f'{name} = {var:.4f}')

    fig, ax = plt.subplots()
    for i in np.where(y)[0]:
        ax.plot(y, '--', color='grey', lw=0.5)
    ax.plot(obs)

    print(f'Firing rate : {y.mean()/Delta : .2f} Hz')

def gen_test_P():
    rng = np.random.default_rng()
    alpha = 0.5*rng.random() + 0.5
    F = 100*rng.random() + 20
    omega = 2*np.pi*F/1000
    R = rot(omega)
    sigma = 1e-3
    P = solve_discrete_lyapunov(alpha*R, sigma*np.eye(2))
    P_hand = (sigma/(1 - np.power(alpha, 2)))*np.eye(2)
    assert np.allclose(P, P_hand)
    np.testing.assert_allclose(np.round(P, 10), P_hand)
    return P
