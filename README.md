# Spike_EM
EM Algorithm to estimate the latent model parameters of a spike train modeled as a Poisson point process.


## Observation Model
We are using a point process model as the observation model with the following Conditional Intensity Function to represent the likelihood of obsrving a spike at time $k\Delta$.
```math
\begin{aligned}
    \lambda_c (k\Delta) &= \exp(\mu_c + x_k) \\
\end{aligned}
```
Here $x_k$ is the real part of the latent vector.

## Latent Model
We define the latent model as an autoregressive model AR(1) where the variable is transformed by a rotation matrix $R(\omega)$.
 ```math
\begin{align*}
\mathbf{x_t} = \alpha R(\omega)\mathbf{x_{t-1}} + \mathbf{u_t} \qquad \forall t = 1...T
\end{align*}
```
Where, $x_t$ is a 2-d vector representing the hidden states that affect a neuron's activity. 

## testing
- Testing Functions: Use to double check implementation of jacobian, hessian and KF
  - check_gradient
  - test_max_L
  - test_KF
  - Use gen_params() for randomized input parameters. 
- Kalman Filter & Fixed Interval Smoother
  - Single ocillation implementation complete
- EM
  - E-step (<b> np.dot </b> is problematic )
  
  
