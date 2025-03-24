import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EM:

    def rot(self, w): 
        '''Rotation matrix for a given angle.
        
        Inputs
        ------
        w : float, Angle in radians (scalar).
        
        Returns
        -------
        ndarray of shape (2, 2), 2x2 rotation matrix.
        '''
        return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]], dtype=np.float64)

    def likelihood(self, x, xm, Pinv, dN, mu_c, beta):
        '''
        Computes the negative log-likelihood for the given parameters.
        
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu_c : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        float, Negative log-likelihood value.
        '''
        dot2 = (-1/2)*np.linalg.multi_dot([(x - xm).T, Pinv, x - xm])
        L = dN*(mu_c + beta*x[0]) - np.exp(mu_c + beta*x[0]) + dot2
        return -L
    
    def jacobian(self, x, xm, Pinv, dN, mu_c, beta):
        '''
        Computes the Jacobian (gradient) of the likelihood function.
        
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu_c : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        ndarray of shape (2,), Jacobian vector.
        '''
        jac = -np.dot(Pinv, x - xm)
        jac[0] += (dN - np.exp(mu_c + beta*x[0]))*beta
        return  -jac
    
    def hessian(self, x, xm, Pinv, dN, mu_c, beta):
        '''
        Computes the Hessian (second derivative) of the likelihood function.
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu_c : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        ndarray of shape (2,2), 2x2 Hessian matrix.
        '''
        hess = -Pinv.copy()
        hess[0,0] -= np.exp(mu_c + beta*x[0])*np.power(beta, 2)
        return -hess

    def max_L(self, x0, xp_k, Pinv_k, y, mu_c, beta, method='trust-krylov'):
        '''
        Maximizes the likelihood function to estimate the state.
        
        Inputs
        ------
        x0 : ndarray of shape (2,), Initial estimate of state.
        xp_k : ndarray of shape (2,), Predicted state.
        Pinv_k : ndarray of shape (2, 2), Inverse of predicted covariance matrix.
        y : float, Observation data.
        mu_c : float, Mean parameter.
        beta : float, Scaling parameter.
        method : str, optional, Optimization method, default is 'trust-krylov'.
        
        Returns
        -------
        OptimizeResult, Result of the optimization. (From scipy.minimze)
        '''
        return minimize(self.likelihood, x0, args=(xp_k, Pinv_k, y, mu_c, beta), 
                           jac=self.jacobian, hess=self.hessian, method = method)
    
    def KF_step(self, x_m_k1, P_m_k1, alpha, sigma, R, mu, beta, y_k, method='trust-krylov'):
        '''
        Performs a single recursive step of the Kalman filter.
        
        Inputs
        ------
        x_m_k1 : ndarray of shape (2,), Previous posterior/measurement state estimate.
        P_m_k1 : ndarray of shape (2, 2), Previous posterior/measurement covariance matrix.
        alpha : float, State transition scaling factor (parameter).
        sigma : float, Process noise variance (parameter).
        R : ndarray of shape (2, 2), Rotation matrix.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        y_k : float, Current observation.
        method : str, optional, Optimization method, default is 'trust-krylov'.
        
        Returns
        -------
        tuple
            xp_k : ndarray of shape (2,), Predicted state.
            xm_k : ndarray of shape (2,), Updated state estimate.
            Pp_k : ndarray of shape (2, 2), Predicted covariance.
            Pm_k : ndarray of shape (2, 2), Updated covariance estimate.
        '''
        #Prediction Step
        xp_k = alpha * np.dot(R, x_m_k1)
        Pp_k = np.power(alpha, 2) * np.linalg.multi_dot([R, P_m_k1, R.T]) + sigma*np.eye(2)
        
        Pinv_k = np.linalg.pinv(Pp_k)
    
        # Measurement step : maximizing likelihood function
        # Using x_{k|k-1} as initial estimate for minimize function
        res = self.max_L(x_m_k1, xp_k, Pinv_k, y_k, mu, beta, method=method)
        
        if not res.success:
            print("Estimation of $x_{m, k}$  failed to converge : " + res.message)
            raise AssertionError("Estimation of $x_{m, k}$  failed to converge : " + res.message)
    
        Pm_k = np.linalg.pinv(res.hess) # Negative not required because hessian function minimizes the "negative likelihood".
        return xp_k, res.x, Pp_k, Pm_k

    def backward_pass(self, alpha, R, x_p, x_m, P_p, P_m, K):
        '''
            Performs the backward smoothing pass for state estimation.
            
            Inputs
            ------
            alpha : float, State transition scaling factor.
            R : ndarray of shape (2, 2), Rotation matrix.
            x_p : ndarray of shape (2, K), Predicted states.
            x_m : ndarray of shape (2, K), Filtered (posterior) states.
            P_p : ndarray of shape (2, 2, K), Predicted covariance matrices.
            P_m : ndarray of shape (2, 2, K), Filtered (posterior) covariance matrices.
            K : int, Number of time steps.
            
            Returns
            -------
            tuple
                x_b : ndarray of shape (2, K), Smoothed states.
                P_b : ndarray of shape (2, 2, K), Smoothed covariance matrices.
                AG : ndarray of shape (2, 2, K), Smoothing gain matrices.
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

    def KFilter(self, x0, y, mu, beta, sigma, alpha, F, Fs, K=None):
        '''
            Runs the Kalman filter with forward and backward passes.
        
            Inputs
            ------
            x0 : ndarray of shape (2,), Initial state estimate.
            y : ndarray of shape (K,), Observed data.
            mu : float, Mean parameter.
            beta : float, Scaling parameter.
            sigma : float, Process noise variance.
            alpha : float, State transition scaling factor.
            F : float, Frequency parameter.
            Fs : float, Sampling frequency.
            K : int, optional, Number of time steps, inferred from y if not provided.
            
            Returns
            -------
            tuple
                x_p : ndarray of shape (2, K), Predicted states.
                x_m : ndarray of shape (2, K), Filtered (posterior) states.
                x_b : ndarray of shape (2, K), Smoothed states.
                P_p : ndarray of shape (2, 2, K), Predicted covariance matrices.
                P_m : ndarray of shape (2, 2, K), Filtered (posterior) covariance matrices.
                P_b : ndarray of shape (2, 2, K), Smoothed covariance matrices.
                AG : ndarray of shape (2, 2, K), Smoothing gain matrices.
        '''
        if not K:
            K = y.shape[1]
        
        omega = 2*np.pi*F/Fs
        R = self.rot(omega)
        
        x_p = np.zeros((2,K))
        x_m = np.zeros((2,K))
        
        P_p = np.zeros((2,2,K))
        P_m = np.zeros((2,2,K))
    
        P_m[:, :, 0] = (sigma/(1 - np.power(alpha, 2)))*np.eye(2) #steady state calculation of P_m as an estimate of P_0|0
        x_m[:, 0] = x0.copy()
    
        #Forward Pass : Kalman Filter
        for k in range(1, K):
            xp_k, xm_k, Pp_k, Pm_k = self.KF_step(x_m[:, k-1], P_m[:, :, k-1], 
                                                     alpha, sigma, R, mu, beta, y[k], method='trust-krylov')
            x_p[:, k], P_p[:, :, k], x_m[:, k], P_m[:, :, k] = xp_k, Pp_k, xm_k, Pm_k
    
        #Backward Pass
        x_b, P_b, AG = self.backward_pass(alpha, R, x_p, x_m, P_p, P_m, K)
        
        return x_p, x_m, x_b, P_p, P_m, P_b, AG

    def E_step(self, x_b, P_b, AG, K, beta):
        """
            Expectation step of the EM algorithm. Computes intermediate statistics A, B, C, and expected spike count.
        
            Inputs
            ------
            x_b : ndarray, shape (2, K), Smoothed state estimates from the backward pass.
            P_b : ndarray, shape (2, 2, K), Smoothed state covariance estimates.
            AG : ndarray, shape (2, 2, K-1), Gain matrices from the backward pass.
            K : int, Number of time steps.
            beta : float, Parameter for the intensity function.
        
            Returns
            -------
            A : ndarray, shape (2, 2), Summation of smoothed covariances and state estimates.
            B : ndarray, shape (2, 2), Cross-covariance matrix for consecutive time steps.
            C : ndarray, shape (2, 2), Summation of smoothed covariance estimates for all time steps.
            mu_exp : float, Expected value of the intensity function.
        """
        A = P_b[:, :, :-1].sum(axis=-1) + np.dot(x_b[:, :-1], x_b[:, :-1].T)
        B = np.matmul(np.transpose(P_b[:, :, 1:], axes=(2,0,1)), np.transpose(AG[:, :, :-1], axes=[2,1,0])).sum(axis=0)  
        B += np.dot(x_b[:, 1:], x_b[:, :-1].T)
        C = P_b[:, :, 1:].sum(axis=-1) + np.dot(x_b[:, 1:], x_b[:, 1:].T)
    
        mu_exp = np.exp(beta*x_b[0, :] + np.power(beta,2)*P_b[0,0, : ]/2).sum()
        return A, B, C, mu_exp

    def M_step(self, A, B, C, mu_exp, dN, K):
        """
            Maximization step of the EM algorithm. Updates model parameters.
        
            Inputs
            ------
            A : ndarray, shape (2, 2), Summation of smoothed covariances and state estimates.
            B : ndarray, shape (2, 2), Cross-covariance matrix for consecutive time steps.
            C : ndarray, shape (2, 2), Summation of smoothed covariance estimates for all time steps.
            mu_exp : float, Expected value of the intensity function.
            dN : ndarray, shape (K,), Observed spike counts or event occurrences.
            K : int, Number of time steps.
        
            Returns
            -------
            mu_h : float, Updated intensity function parameter.
            alpha_h : float, Updated state transition scaling factor.
            omega_h : float, Updated rotation angle parameter.
            sigma_h : float, Updated process noise variance.
        """
        omega_h = np.arctan2(B[1,0] - B[0,1], np.trace(B))
        alpha_h = (np.trace(B)*np.cos(omega_h) + (B[1,0] - B[0,1])*np.sin(omega_h)) / np.trace(A)
        sigma_h = (np.trace(C) - np.power(alpha_h, 2)*np.trace(A))/(2*K)
        
        mu_h = np.log(np.sum(dN)) - np.log(mu_exp)
    
        return mu_h, alpha_h, omega_h, sigma_h

    def EM_debug(self, y, x0, mu0, beta, sigma0, alpha0, F0, Fs, iters = 100, method='trust-krylov', verbose=1):
        """
            Runs the Expectation-Maximization (EM) algorithm to estimate model parameters. Logs all intermediate KF outputs and E-step estimates as well as errors due to non-linear estimation of x_k by scipy.minimize.
        
            Inputs
            ------
            y : ndarray, shape (K,), Observed spiking activity.
            x0 : ndarray, shape (2,), Initial state estimate.
            mu0 : float, Initial intensity function parameter.
            beta : float, Parameter for the intensity function.
            sigma0 : float, Initial process noise variance.
            alpha0 : float, Initial state transition scaling factor.
            F0 : float, Initial frequency estimate.
            Fs : float, Sampling frequency.
            iters : int, default=100, Maximum number of EM iterations.
            method : str, default='trust-krylov', Optimization method used for likelihood maximization.
            verbose : int, default=1, Verbosity, decides if it prints after each iteration to track progress.
        
            Returns
            -------
            params : DataFrame, Estimated parameters over iterations.
            E_steps : dict, Dictionary containing intermediate results from the E-step.
            final_params : list, Final estimated parameters [mu_h, alpha_h, F0, omega_h, sigma_h].
            error_log : DataFrame, Log of any errors encountered during EM iterations.
        """
        K = len(y)
        
        omega0 = 2*np.pi*F0/Fs
        
        params = pd.DataFrame(columns=['$\\mu$', '$\\alpha$', '$\\omega$', '$\\sigma^2$'])
        params.loc[0, :] = [mu0, alpha0, omega0, sigma0]
    
        error_log = pd.DataFrame(columns = ['n_iter', 'x0', 'mu_h', 'alpha_h', 'omega_h', 'sigma_h', 'msg'])
        
        E_steps = {'x_p': [],'x_m': [],'x_b': [],'P_p': [],'P_m': [],'P_b': [], 'BG':[], 'A':[], 'B':[], 'C':[]}
        
        for itr in range(1, iters+1):
            if verbose >= 1:
                print(itr, end=' ')
    
            try:
                # Filtering to get x and P estimates
                x_pred, x_meas, x_smooth, P_pred, P_meas, P_smooth, BG = self.KFilter(x0, y, mu0, beta, sigma0, alpha0, F0, Fs, K)
                
                for k,v in zip(['x_p','x_m','x_b','P_p','P_m','P_b', 'BG'], 
                               [x_pred, x_meas, x_smooth, P_pred, P_meas, P_smooth, BG]):
                    E_steps[k].append(v)
                
                # Maximizing to update param values
                # E-step
                A,B,C, mu_exp = self.E_step(x_smooth, P_smooth, BG, K, beta)
                
                for k,v in zip(['A', 'B', 'C'], [A, B, C]):
                    E_steps[k].append(v)
        
                # M-step
                mu_h, alpha_h, omega_h, sigma_h = self.M_step(A, B, C, mu_exp, y, K)
                params.loc[itr, :] = [mu_h, alpha_h, omega_h, sigma_h]
        
                #Update for next iter
                F0 = omega_h/(2*np.pi/Fs) # since filter takes in F and not omega
                mu0, alpha0, omega0, sigma0 = mu_h, alpha_h, omega_h, sigma_h
                
            except AssertionError as err:
                error_log.loc[error_log.shape[0], :] = [itr, x0, mu0, alpha0, omega0, sigma0, err.__str__()]
    
        return params, E_steps, [mu_h, alpha_h, F0, omega_h, sigma_h], error_log

    def run_EM(self, y, x0, mu0, beta, sigma0, alpha0, F0, Fs, iters = 100, method='trust-krylov'):
        """
            Runs the Expectation-Maximization (EM) algorithm to estimate model parameters. 
        
            Inputs
            ------
            y : ndarray, shape (K,), Observed spiking activity.
            x0 : ndarray, shape (2,), Initial state estimate.
            mu0 : float, Initial intensity function parameter.
            beta : float, Parameter for the intensity function.
            sigma0 : float, Initial process noise variance.
            alpha0 : float, Initial state transition scaling factor.
            F0 : float, Initial frequency estimate.
            Fs : float, Sampling frequency.
            iters : int, default=100, Maximum number of EM iterations.
            method : str, default='trust-krylov', Optimization method used for likelihood maximization.
        
            Returns
            -------
            params : DataFrame, Estimated parameters over iterations.
            final_params : list, Final estimated parameters [mu_h, alpha_h, F0, omega_h, sigma_h].
            
        """
        K = len(y)
        
        omega0 = 2*np.pi*F0/Fs
        
        params = pd.DataFrame(columns=['$\\mu$', '$\\alpha$', '$\\omega$', '$\\sigma^2$'])
        params.loc[0, :] = [mu0, alpha0, omega0, sigma0]
    
        for itr in range(1, iters+1):
            # Filtering to get x and P estimates
            x_pred, x_meas, x_smooth, P_pred, P_meas, P_smooth, BG = self.KFilter(x0, y, mu0, beta, sigma0, alpha0, F0, Fs, K)
            # E-step
            A,B,C, mu_exp = self.E_step(x_smooth, P_smooth, BG, K, beta)  
            # M-step
            mu_h, alpha_h, omega_h, sigma_h = self.M_step(A, B, C, mu_exp, y, K)
            params.loc[itr, :] = [mu_h, alpha_h, omega_h, sigma_h]    
            #Update for next iter
            F0 = omega_h/(2*np.pi/Fs) # since filter takes in F and not omega
            mu0, alpha0, omega0, sigma0 = mu_h, alpha_h, omega_h, sigma_h

        return params, [mu_h, alpha_h, F0, omega_h, sigma_h]
    