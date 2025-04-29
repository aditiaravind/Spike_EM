import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import factorial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("default", category=RuntimeWarning)


class EMEstimator(BaseEstimator):
    def __init__(self, x0, mu0, alpha0, sigma0, F0, beta, Fs, omega0=None, gamma=10, method='trust-krylov', max_iter=100, tol=1e-4, am=None, bm=None):
        """Initializes the EMEstimator with user-defined or default hyperparameters.
    
        Parameters
        ----------
        x0 : ndarray of shape (2,)
            Initial state estimate (x0).
        mu0 : float
            Initial value for the parameter (mu).
        alpha0 : float
            Initial value for the state transition scaling factor (alpha).
        sigma0 : float
            Initial value for the process noise variance (sigma).
        F0 : float
            Initial value for frequency (Hz).
        beta : float
            Scaling parameter for the state-dependent intensity.
        gamma : float
            Regularization parameter that adds an exponential prior on sigma to prevent parameter explosion
        Fs : float
            Sampling frequency. Used only to calculate omega
        omega0 : float, optional
            Initial angular frequency (rad/s). If None, computed from F0 and Fs.
        method : str, optional
            Optimization method used in likelihood maximization. Default is 'trust-krylov'.
        max_iter : int, optional
            Maximum number of iterations for the EM algorithm. Default is 100.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.x0 = x0
        self.mu0 = mu0
        self.alpha0 = alpha0
        self.sigma0 = sigma0
        self.Fs = Fs
        self.F0 = F0
        if not omega0:
            self.omega0 = 2*np.pi*F0/Fs
        else:
            self.omega0 = omega0
        self.method = method
        self.beta = beta
        self.gamma = gamma
        self.am = am
        self.bm = bm
        


    def fit(self, X, y, log_file=None):
        """Fits the EM estimator to observed data using the Expectation-Maximization algorithm.
    
        Parameters
        ----------
        X : ndarray of shape (n,K)
            dummy array
        y : ndarray of shape (K,)
            Observed data (e.g., spike counts or event times).
        log_file : str, optional
            Path to a CSV file for saving parameter log during iterations. If None, no log is saved.
    
        Returns
        -------
        self : object
            Fitted estimator.
        """
        params, E_steps, [mu_h, alpha_h, F_h, omega_h, sigma_h], error_log = self._em_algorithm(y, 
                                                                                                x0 = self.x0, mu0 = self.mu0, beta = self.beta, sigma0 = self.sigma0, alpha0 = self.alpha0, F0 = self.F0, max_iter = self.max_iter, log_file=log_file)
        self.mu_ = mu_h
        self.alpha_ = alpha_h
        self.F_ = F_h
        self.omega_ = omega_h
        self.sigma_ = sigma_h
        self._param_history = params
        self._error_log = error_log
        for key, value in E_steps.items():
            setattr(self, f"_{key}_history", np.array(value))


    def predict(self, y):
        """Returns the final fitted parameters after EM convergence.
    
        Parameters
        ----------
        X : ndarray of shape (n, K)
            Dummy input for compatibility. Ignored during execution.
        y : ndarray of shape (K,)
            Observed data (e.g., spike counts or event times).
    
        Returns
        -------
        dict
            Dictionary containing fitted values of 'mu', 'alpha', 'sigma', 'F', and 'omega'.
        """
        check_is_fitted(self, ['mu_', 'alpha_', 'sigma_', 'F_'])     
        x_p, _, _, _ = self._KFilter(x0=self.x0, y = y, mu = self.mu_, beta = self.beta, sigma = self.sigma_, alpha = self.alpha_, F = self.F_)
        return x_p

    
    def score(self, X, y, x0=None, mu=None, sigma=None, alpha=None, F = None, beta = None, order = 3):
        """Scores the model based on the log-likelihood. Uses estimated parameters if not provided.
    
        Parameters
        ----------
        X : ndarray of shape (n, K)
            Dummy input for compatibility. Ignored during execution.
        y : ndarray of shape (K,)
            Observed data.
        
        Returns
        -------
        log_score : float
            Log-likelihood score.
        """
        check_is_fitted(self, ['mu_', 'alpha_', 'sigma_', 'F_'])
        
        
        x0 = self.x0 if x0 is None else x0
        mu = self.mu_ if mu is None else mu
        alpha = self.alpha_ if alpha is None else alpha
        sigma = self.sigma_ if sigma is None else sigma
        F = self.F_ if F is None else F
        beta = self.beta if beta is None else beta
        
        x_p, _, P_p, _ = self._KFilter(x0 = x0, y = y, mu = mu, beta = beta, sigma = sigma, alpha = alpha, F = F)
        log_score = self._log_likelihood(dN = y, mu = mu, x_p = x_p, P_p = P_p, beta = beta, order=order)
        return log_score

    def plot_score_history(self, y, step=10, beta=None, order=3, ret=False):
        """Plots the log-likelihood score history over EM iterations.
    
        Parameters
        ----------
        y : ndarray of shape (K,)
            Observed data.
        step : int, optional
            Interval at which to evaluate and plot the score. Default is 10.
        beta : float, optional
            Scaling parameter. If None, uses self.beta.
        order : int, optional
            Order for the log-likelihood approximation. Default is 3.
        """
        check_is_fitted(self, ['_param_history', '_x_p_history', '_P_p_history'])
    
        if beta is None:
            beta = self.beta
    
        score = []
        iterations = list(range(0, len(self._x_p_history)+1, step))
        
        for itr in iterations:
            mu_itr = self._param_history.loc[itr, 'mu_']
            alpha_itr = self._param_history.loc[itr, 'alpha_']
            F_itr = self._param_history.loc[itr, 'F_']
            omega_itr = self._param_history.loc[itr, 'omega_']
            sigma_itr = self._param_history.loc[itr, 'sigma_']
            
            xp_itr = self._x_p_history[itr-1]
            Pp_itr = self._P_p_history[itr-1]
    
            score_val = self._log_likelihood(y, mu_itr, xp_itr, Pp_itr, beta, order=order)
            score.append(score_val)
    
        plt.figure(figsize=(6, 4))
        plt.plot(iterations, score, marker='o')
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-likelihood")
        plt.title("Score (Log-likelihood) History")
        plt.tight_layout()
        plt.show()

        if ret:
            return score


    def plot_param_history(self, true_params=None):
        """Plots the parameter evolution history over EM iterations for selected parameters.
    
        Parameters
        ----------
        true_params : list or tuple of float, optional
            Ground truth values of parameters [mu, alpha, omega, sigma] to be plotted for comparison.
            If None, only estimated and initial values are shown.
    
        Notes
        -----
        Requires the estimator to be fitted. Uses `self._param_history` and `self.omega0`, etc.
        """
        check_is_fitted(self, ['_param_history'])
    
        param_names = ['mu_', 'alpha_', 'F_', 'sigma_']
        initial_params = [self.mu0, self.alpha0, self.F0, self.sigma0]
    
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        
        for ix, (name, ax, init_val) in enumerate(zip(param_names, axs.ravel(), initial_params)):
            ax.plot(self._param_history[name], label='Estimated', color='C0')
            ax.axhline(init_val, color='grey', linestyle='--', label='Init')
            
            if true_params is not None:
                ax.axhline(true_params[ix], color='black', linestyle='-.', label='True')
    
            ax.set_title(name)
            ax.legend()
    
        fig.supxlabel("EM Iteration")
        fig.tight_layout()
        plt.show()
        
    def _k_likelihood(self, dN_k, mu, xp_k, Pp_k, beta, order = 3):
        """Computes the approximate log-likelihood contribution for a single time point.
        
        Parameters
        ----------
        dN_k : float
            Observed count or event at time k.
        mu : float
            Mean parameter.
        xp_k : ndarray of shape (2,)
            Predicted state vector at time k.
        Pp_k : ndarray of shape (2, 2)
            Predicted covariance matrix at time k.
        beta : float
            Scaling parameter.
        order : int, optional
            Order of the Poisson expansion for log-likelihood approximation. Default is 3.
        
        Returns
        -------
        logL_k : float
            Approximate log-likelihood for the time step.
        """
        ### DO LSE as kxn array after subtracting max(kxn) !!!
        L = 0
        for n in range(0, order+1):
            exp_term = (dN_k + n)*(mu + xp_k[0] + (dN_k + n)*Pp_k[0,0]/2)
            L += ((-1)**n/factorial(n))*np.exp(exp_term)
        logL_k = np.log(L)
        return logL_k
        
    def _log_likelihood(self, dN, mu, x_p, P_p, beta, order = 3):
        """Computes the total log-likelihood over all time steps.
    
        Parameters
        ----------
        dN : ndarray of shape (K,)
            Observed spike counts or event data.
        mu : float
            Mean parameter.
        x_p : ndarray of shape (2, K)
            Predicted states from the Kalman filter.
        P_p : ndarray of shape (2, 2, K)
            Predicted covariance matrices from the Kalman filter.
        beta : float
            Scaling parameter.
        order : int, optional
            Order of the Poisson expansion for log-likelihood approximation. Default is 3.
    
        Returns
        -------
        logL : float
            Total log-likelihood across time steps.
        """
        K = dN.shape[-1]
        logL = 0
        for k in range(K):
            logL += self._k_likelihood(dN[k], mu, x_p[:, k], P_p[..., k], beta, order = order)
        return logL


    def _rot(self, w):
        """Rotation matrix for a given angle.
        
        Inputs
        ------
        w : float, Angle in radians (scalar).
        
        Returns
        -------
        Rw : ndarray of shape (2, 2), 2x2 rotation matrix.
        """
        Rw = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]], dtype=np.float64)
        return Rw
    
    def _rt_matrix(self, array):
        """Returns rt(array) = array[1,0] - array[0,1]
    
        Inputs
        ------
        array : ndarray, shape (2, 2)

        Returns
        -------
        rt : float, rt of matrix. 
        """
        assert array.shape == (2,2)
        rt = array[1,0] - array[0,1]
        return rt

    def _likelihood(self, x, xm, Pinv, dN, mu, beta):
        """Computes the negative log-likelihood of p(x_k|H(k))
        
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        L : float, Negative log-likelihood value.
        """
        dot2 = (-1/2)*np.linalg.multi_dot([(x - xm).T, Pinv, x - xm])
        L = dN*(mu + beta*x[0]) - np.exp(mu + beta*x[0]) + dot2
        return -L
        # dot2 = (1/2)*np.linalg.multi_dot([(x - xm).T, Pinv, x - xm])
        # L = dot2 - dN*(mu + beta*x[0]) - np.exp(mu + beta*x[0])
        # return L
    
    def _jacobian(self, x, xm, Pinv, dN, mu, beta):
        """Computes the Jacobian (gradient) of the likelihood function of p(x_k|H(k)).
        
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        jac : ndarray of shape (2,), Jacobian vector.
        """
        jac = -np.dot(Pinv, x - xm)
        jac[0] += (dN - np.exp(mu + beta*x[0]))*beta
        return  -jac
        # jac = np.dot(Pinv, x - xm)
        # jac[0] -= (dN - np.exp(mu + beta*x[0]))*beta
        # return  jac
    
    def _hessian(self, x, xm, Pinv, dN, mu, beta):
        """Computes the Hessian (second derivative) of the likelihood function of p(x_k|H(k)).
        Inputs
        ------
        x : ndarray of shape (2,), Current state estimate.
        xm : ndarray of shape (2,), Mean of the prior distribution.
        Pinv : ndarray of shape (2, 2), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        hess : ndarray of shape (2,2), 2x2 Hessian matrix.
        """
        hess = -Pinv.copy()
        hess[0,0] -= np.exp(mu + beta*x[0])*np.power(beta, 2)
        return -hess
        # hess = Pinv.copy()
        # hess[0,0] += np.exp(mu + beta*x[0])*np.power(beta, 2)
        # return hess

    def _max_L(self, x0, xp_k, Pinv_k, y, mu, beta):
        """Maximizes the likelihood function to estimate the state.
        
        Inputs
        ------
        x0 : ndarray of shape (2,), Initial estimate of state.
        xp_k : ndarray of shape (2,), Predicted state.
        Pinv_k : ndarray of shape (2, 2), Inverse of predicted covariance matrix.
        y : float, Observation data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        
        Returns
        -------
        res : OptimizeResult, Result of the optimization. (From scipy.minimze)
        """
        res = minimize(self._likelihood, x0, args=(xp_k, Pinv_k, y, mu, beta), 
                           jac=self._jacobian, hess=self._hessian, method = self.method)
        return res
    
    def _KF_step(self, x_m_k1, P_m_k1, alpha, sigma, R, mu, beta, y_k):
        """Performs a single recursive step of the Kalman filter.
        
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
        
        Returns
        -------
        tuple
            xp_k : ndarray of shape (2,), Predicted state.
            xm_k : ndarray of shape (2,), Updated state estimate.
            Pp_k : ndarray of shape (2, 2), Predicted covariance.
            Pm_k : ndarray of shape (2, 2), Updated covariance estimate.
        """
        #Prediction Step
        xp_k = alpha * np.dot(R, x_m_k1)
        Pp_k = np.power(alpha, 2) * np.linalg.multi_dot([R, P_m_k1, R.T]) + sigma*np.eye(2)
        
        Pinv_k = np.linalg.pinv(Pp_k)
    
        # Measurement step : maximizing likelihood function
        # Using x_{k|k-1} as initial estimate for minimize function
        res = self._max_L(x_m_k1, xp_k, Pinv_k, y_k, mu, beta)
        
        if not res.success:
            print("Estimation of $x_{m, k}$  failed to converge : " + res.message)
            raise AssertionError("Estimation of $x_{m, k}$  failed to converge : " + res.message)
        xm_k = res.x
        Pm_k = np.linalg.pinv(res.hess) # Negative not required because hessian function minimizes the "negative likelihood".
        return xp_k, xm_k, Pp_k, Pm_k

    def _KFilter(self, x0, y, mu, beta, sigma, alpha, F, K=None):
        """Runs the Kalman filter for the forward pass of the E-step.
    
        Inputs
        ------
        x0 : ndarray of shape (2,), Initial state estimate.
        y : ndarray of shape (K,), Observed data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        sigma : float, Process noise variance.
        alpha : float, State transition scaling factor.
        F : float, Frequency parameter.
        K : int, optional, Number of time steps, inferred from y if not provided.
        
        Returns
        -------
        tuple
            x_p : ndarray of shape (2, K), Predicted states.
            x_m : ndarray of shape (2, K), Filtered (posterior) states.
            P_p : ndarray of shape (2, 2, K), Predicted covariance matrices.
            P_m : ndarray of shape (2, 2, K), Filtered (posterior) covariance matrices.
        """
        if not K:
            K = y.shape[-1]
        
        omega = 2*np.pi*F/self.Fs
        R = self._rot(omega)
        
        x_p = np.zeros((2,K))
        x_m = np.zeros((2,K))
        
        P_p = np.zeros((2,2,K))
        P_m = np.zeros((2,2,K))
    
        P_m[:, :, 0] = (sigma/(1 - np.power(alpha, 2)))*np.eye(2)
        x_m[:, 0] = x0.copy()
    
        #Forward Pass : Kalman Filter
        for k in range(1, K):
            xp_k, xm_k, Pp_k, Pm_k = self._KF_step(x_m[:, k-1], P_m[:, :, k-1], 
                                                     alpha = alpha, sigma = sigma, R=R, mu = mu, beta = beta, y_k = y[k])
            x_p[:, k], P_p[:, :, k], x_m[:, k], P_m[:, :, k] = xp_k, Pp_k, xm_k, Pm_k
    
        
        return x_p, x_m, P_p, P_m
    
    def _backward_pass(self, alpha, R, x_p, x_m, P_p, P_m, K):
        """Performs the backward smoothing pass for state estimation.
        
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
            back_gains : ndarray of shape (2, 2, K), Smoothing gain matrices.
        """
        back_gains = np.zeros((2,2,K))
        x_b = np.zeros((2,K))
        P_b = np.zeros((2,2,K))
        x_b[:, -1] = x_m[:, -1].copy()
        P_b[:, :, -1] = P_m[:, :, -1].copy()
        for k in range(K-1, 0, -1):
            A_k1 = np.linalg.multi_dot([P_m[:, :, k-1], alpha*R.T, np.linalg.pinv(P_p[:, :, k])])
            back_gains[:, :, k-1] = A_k1
    
            x_b[:, k-1] = x_m[:, k-1] + np.dot(A_k1, x_b[:, k] - x_p[:, k])
            P_b[:, :, k-1] = P_m[:, :, k-1] + np.linalg.multi_dot([A_k1, P_b[:, :, k] - P_p[:, :, k], A_k1.T])
            
        return x_b, P_b, back_gains
    


    def _E_step(self, x0, y, mu, beta, sigma, alpha, F, K=None):
        """Expectation step of the EM algorithm. Runs the Kalman Filter and Backward pass. Computes intermediate statistics A, B, C, and expected spike count that are used in the M-step for maximizing the parameter estimates.
    
        Inputs
        ------
        x0 : ndarray of shape (2,), Initial state estimate.
        y : ndarray of shape (K,), Observed data.
        mu : float, Mean parameter.
        beta : float, Scaling parameter.
        sigma : float, Process noise variance.
        alpha : float, State transition scaling factor.
        F : float, Frequency parameter.
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
            back_gains : ndarray of shape (2, 2, K), Smoothing gain matrices.
            A : ndarray, shape (2, 2), Summation of smoothed covariances and state estimates.
            B : ndarray, shape (2, 2), Cross-covariance matrix for consecutive time steps.
            C : ndarray, shape (2, 2), Summation of smoothed covariance estimates for all time steps.
            mu_sum : float, Value of the intensity function used to maximize mu parameter.
        """
    
        #Forward Pass
        x_p, x_m, P_p, P_m = self._KFilter(x0 = x0, y = y, mu = mu, beta = beta, sigma = sigma, alpha = alpha, F = F, K = K)
            
        omega = 2*np.pi*F/self.Fs
        R = self._rot(omega)
        
        #Backward Pass
        x_b, P_b, back_gains = self._backward_pass(alpha=alpha, R=R, x_p=x_p, x_m=x_m, P_p=P_p, P_m=P_m, K=K) 
        
        A = P_b[:, :, :-1].sum(axis=-1) + np.dot(x_b[:, :-1], x_b[:, :-1].T)
        B = np.matmul(np.transpose(P_b[:, :, 1:], axes=(2,0,1)), np.transpose(back_gains[:, :, :-1], axes=[2,1,0])).sum(axis=0)
        B += np.dot(x_b[:, 1:], x_b[:, :-1].T)
        C = P_b[:, :, 1:].sum(axis=-1) + np.dot(x_b[:, 1:], x_b[:, 1:].T)

        #Mu update term -> estimate of expinential term from x_b and P_b
        mu_sum = beta*x_b[0, :] + np.power(beta,2)*P_b[0,0, : ]/2
        return x_p, x_m, x_b, P_p, P_m, P_b, back_gains, A, B, C, mu_sum
    
    def _mu_prior_init(self, dN, K):
        """Initialize hyperparameters for mu priors.
        
        Inputs
        ------
        dN : ndarray, shape (K,), Observed spike counts or event occurrences.
        K : int, Number of time steps.
        
        Returns
        -------
        am : float, Hyperparameter alpha_m that is analogous to 'adding am observed spikes' to the estimate of mu.
        bm : float, Hyperparameter beta_m that is analogous to 'adding am observed spikes within interval bm' to the estimate of mu.
        """
        am, bm = dN.sum(), K
        self.am = am
        self.bm = bm
        return am, bm

        
    def _M_step(self, A, B, C, mu_sum, dN, K=None):
        """Maximization step of the EM algorithm. Updates model parameters.

        Inputs
        ------
        A : ndarray, shape (2, 2), Summation of smoothed covariances and state estimates.
        B : ndarray, shape (2, 2), Cross-covariance matrix for consecutive time steps.
        C : ndarray, shape (2, 2), Summation of smoothed covariance estimates for all time steps.
        mu_exp : float, Expected value of the intensity function.
        dN : ndarray, shape (K,), Observed spike counts or event occurrences.
        K : int, (optional) Number of time steps.

        Returns
        -------
        mu_h : float, Updated intensity function parameter.
        alpha_h : float, Updated state transition scaling factor.
        omega_h : float, Updated rotation angle parameter.
        sigma_h : float, Updated process noise variance.
        """
        if K == None:
            K = dN.shape[-1]
        if (self.am is None) or(self.bm is None):
            am, bm = self._mu_prior_init(dN, K)
        else:
            am, bm = self.am, self.bm
        
        # omega_h = np.arctan2(B[1,0] - B[0,1], np.trace(B))
        omega_h = np.arctan2(self._rt_matrix(B), np.trace(B))
        alpha_h = (np.trace(B)*np.cos(omega_h) + self._rt_matrix(B)*np.sin(omega_h)) / np.trace(A)
        Tr = np.trace(C) - np.power(alpha_h, 2)*np.trace(A)
        sigma_h = (-K + np.sqrt(K**2 + 2*self.gamma*Tr))/(2*self.gamma)
        # mu_h = np.log(np.sum(dN)) - mu_sum.max() - np.log(np.exp(mu_sum - mu_sum.max()).sum())
        mu_h = np.log(np.sum(dN) + am - 1) - mu_sum.max() - np.log(np.exp(mu_sum - mu_sum.max()).sum() + np.exp(np.log(bm) - mu_sum.max())) 
    
        return mu_h, alpha_h, omega_h, sigma_h
        
        
    def _em_algorithm(self, y, x0, mu0, beta, sigma0, alpha0, F0, max_iter = 100, log_file = None):

        K = y.shape[-1]
        
        omega0 = 2*np.pi*F0/self.Fs
        
        params = pd.DataFrame(columns=['mu_', 'alpha_', 'F_', 'omega_', 'sigma_'])
        params.loc[0, :] = [mu0, alpha0, F0, omega0, sigma0]

        error_log = pd.DataFrame(columns = ['n_iter', 'x0', 'mu_', 'alpha_', 'F_', 'omega_', 'sigma_', 'msg'])
        
        E_steps = {'x_p': [],'x_m': [],'x_b': [],'P_p': [],'P_m': [],'P_b': [], 'back_gains':[], 'A':[], 'B':[], 'C':[]}

        for itr in range(1, max_iter+1):
            if np.mod(itr, 10) == 0:
                print(itr, end=' ')
    
            if itr > 510:
                if params.tail(500).diff().iloc[1:].map(lambda x:np.allclose(x, 0)).apply(lambda x: x.all()).any():
                    print('Early stopping criteria met.')
                    break
            try:
                # E_step
                x_p, x_m, x_b, P_p, P_m, P_b, BG, A, B, C, mu_sum = self._E_step(x0 = x0, y = y, mu = mu0, beta = beta, sigma = sigma0, alpha = alpha0, F = F0, K = K)
                
                #Save E-step vars
                for k,v in zip(['x_p','x_m','x_b','P_p','P_m','P_b', 'back_gains', 'A', 'B', 'C'],
                               [x_p, x_m, x_b, P_p, P_m, P_b, BG, A, B, C]):
                    E_steps[k].append(v)
                    
                # M-step
                mu_h, alpha_h, omega_h, sigma_h = self._M_step(A, B, C, mu_sum, y, K)
                

                #Save M-step Updated parameter values
                F_h = omega_h/(2*np.pi/self.Fs) # since filter takes in F and not omega
                params.loc[itr, :] = [mu_h, alpha_h, F_h, omega_h, sigma_h]
                if log_file:
                    params.to_csv(log_file)

                #Update for next iter
                mu0, alpha0, F0, omega0, sigma0 = mu_h, alpha_h, F_h, omega_h, sigma_h
                # x0 = x_b[:, 0]
                
            except AssertionError as err:
                error_log.loc[error_log.shape[0], :] = [itr, x0, mu0, alpha0, F0, omega0, sigma0, err.__str__()]
    
        return params, E_steps, [mu_h, alpha_h, F_h, omega_h, sigma_h], error_log
