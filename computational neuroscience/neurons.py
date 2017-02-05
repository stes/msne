""" A collection of neuron models, implemented in numpy

Calls to all models are identical and require a set of timesteps (which do not necessary
need to be equally spaced!) and a matrix or vector holding input currents.
It is possible to simulate multiple neurons in parallel, in this case simply pass a matrix
shaped ``nb_timesteps, nb_neurons`` as the input current.

By default, the first timestep is initialized by the steady state of the model assuming
a current ``I[0,:] = 0``, so it might benefit the simulation to have the first non-zero
entries at the second timestep or later.
"""

import numpy as np

def hodgkin_huxley(tt, I, u_off=-65.):
    """ Simulation of Hodkin-Huxley neuron model
    
    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    u_off : float, resting membrane potential
    """
    
    # Sodium: m³ is probability, h gating variable
    α_m = lambda u: (2.5 - 0.1*(u - u_off)) / (np.exp(2.5 - 0.1*(u - u_off)) - 1)
    β_m  = lambda u : 4*np.exp(-(u - u_off)/18)
    α_h = lambda u: 0.07 * np.exp(-(u - u_off)/20)
    β_h  = lambda u : 1/(np.exp(3 - 0.1*(u - u_off))+1)
    g_Na = 120
    E_Na = 115 + u_off

    # Potassium: n^4 is the probability (4 activation particles)
    α_n = lambda u: (0.1 - 0.01*(u - u_off)) / (np.exp(1-0.1*(u - u_off))-1)
    β_n  = lambda u : 0.125 * np.exp(-(u - u_off)/80)
    g_K = 36
    E_K = -12 + u_off

    # Everything else is covered by E_L
    g_L = 0.3
    E_L = 10.6 + u_off
    
    # Capacitance controls time constant
    C = 0.5
    
    n,m,h,u = [np.zeros_like(I) for i in range(4)]
    
    # Initialize by steady states
    n[0] = α_n(u_off) / (α_n(u_off) + β_n(u_off))
    m[0] = α_m(u_off) / (α_m(u_off) + β_m(u_off))
    h[0] = α_h(u_off) / (α_h(u_off) + β_h(u_off))
    u[0] = u_off
    
    for t in range(1, len(tt)):
        dt = tt[t] - tt[t-1]
        
        dndt = α_n(u[t-1]) * (1 - n[t-1]) - β_n(u[t-1]) * n[t-1]
        dmdt = α_m(u[t-1]) * (1 - m[t-1]) - β_m(u[t-1]) * m[t-1]
        dhdt = α_h(u[t-1]) * (1 - h[t-1]) - β_h(u[t-1]) * h[t-1]
        dudt = ( g_Na * (E_Na - u[t-1]) * m[t-1]**3 * h[t-1] \
               + g_K  * (E_K - u[t-1]) *  n[t-1]**4 \
               + g_L  * (E_L - u[t-1]) + I[t-1]) / C
        
        n[t] = n[t-1] + dt * dndt
        m[t] = m[t-1] + dt * dmdt
        h[t] = h[t-1] + dt * dhdt
        u[t] = u[t-1] + dt * dudt
        
    
    return u, (n, m, h)

def hodgkin_huxley_simple(tt, I, u_off=-65.):
    """ Simulation of the simplified Hodkin-Huxley neuron model

    In the simplified Hodkin-Huxley model, the sodium gating variable ``m``
    is replaced by its steady state ``α_m / ( α_m +  β_m )`` due to the faster
    dynamics.
    In addition, the gating variable ``h`` for sodium is tied to be ``1-n``,
    according to the fact that sodium channels have high probability to close
    again when most of the potassium channels open during the falling phase of
    the action potential.
    
    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    u_off : float, resting membrane potential
    """
    
    # Sodium: m³ is probability, h gating variable
    α_m = lambda u: (2.5 - 0.1*(u - u_off)) / (np.exp(2.5 - 0.1*(u - u_off)) - 1)
    β_m  = lambda u : 4*np.exp(-(u - u_off)/18)
    g_Na = 120
    E_Na = 115 + u_off

    # Potassium: n^4 is the probability (4 activation particles)
    α_n = lambda u: (0.1 - 0.01*(u - u_off)) / (np.exp(1-0.1*(u - u_off))-1)
    β_n  = lambda u : 0.125 * np.exp(-(u - u_off)/80)
    g_K = 36
    E_K = -12 + u_off

    # Everything else is covered by E_L
    g_L = 0.3
    E_L = 10.6 + u_off
    
    # Capacitance controls time constant
    C = 0.5
    
    n,m,u = [np.zeros_like(I) for i in range(3)]
    
    # Initialize by steady states
    n[0] = α_n(u_off) / (α_n(u_off) + β_n(u_off))
    m[0] = α_m(u_off) / (α_m(u_off) + β_m(u_off))
    u[0] = u_off
    
    for t in range(1, len(tt)):
        dt = tt[t] - tt[t-1]
        
        dndt = α_n(u[t-1]) * (1 - n[t-1]) - β_n(u[t-1]) * n[t-1]
        dudt = (g_Na * (E_Na - u[t-1]) * m[t-1]**3 * (1 - n[t-1]) \
               + g_K * (E_K - u[t-1]) *  n[t-1]**4\
               + g_L * (E_L - u[t-1]) + I[t-1]) / C
        
        n[t] = n[t-1] + dt * dndt
        m[t] = α_m(u[t-1]) / (α_m(u[t-1]) + β_m(u[t-1]))
        u[t] = u[t-1] + dt * dudt
        
    return u, (n, m)


def fitzhugh_nagumo(tt, I, rescale=True):
    """ Simulation of the FitzHugh-Nagumo neuron model
    
    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    """

    
    eps = 0.1
    b0 = 2
    b1 = 1.5
    
    w,u = [np.zeros_like(I) for i in range(2)]
    
    assert b1 > 1, "Starting point ambigous for b1 <= 1!"
    u0 = np.roots([1/3,0,b1 - 1,b0])
    u0 = np.real(u0)[np.isclose(u0,np.real(u0))][0]
    
    u[0] = u0
    w[0] = b0 + b1*u0
    
    for t in range(1, len(tt)):
        dt = tt[t] - tt[t-1]
        
        dwdt = eps * (b0 + b1 * u[t-1] - w[t-1])
        dudt = (u[t-1] - u[t-1]**3 / 3 - w[t-1] + I[t-1] / 10)
        
        w[t] = w[t-1] + dt * dwdt * 3
        u[t] = u[t-1] + dt * dudt * 3
    
    if rescale:
        return u*30-15, (w,)
    else:
        return u, (w,)

def leaky_integrate(tt, I, spike_rate_adaption = True, refractory = False, slow_down = 8/500):
    """ Simulation of the Leaky Integrate and Fire model with optional spike rate adaption

    tt : ndarray, one dimension, contains timepoints
    I  : ndarray, one or two dimensions, contains input current per neuron
    slow_down : factor for scaling down the timesteps in order to be in the same magnitude as
                the HH and FHN models
    """
    
    flatten = False
    if I.ndim == 1:
        I = I[:, np.newaxis]
        flatten = True

    E_L = -65
    E_K = -65
    R_m = 10
    threshold = -50
    u_t = 50
    u_r = -65

    tau = 2 #ms refractory period
    tau_rsa = 100
    delta_g_rsa = 0.002

    tau_r = 2
    delta_g_r = 0.1

    u, g_rsa, g_r = [np.zeros_like(I) for i in range(3)]

    u[0] = u_r
    g_rsa[0] = 0
    g_r[0] = 0

    for t in range(1, len(tt)):
        dt = (tt[t] - tt[t-1]) * slow_down
        
        du = (R_m * I[t-1] 
                + spike_rate_adaption * R_m * g_rsa[t-1] * (E_K - u[t-1])
                + refractory * R_m * g_r[t-1] * (E_K - u[t-1])
                + (E_L - u[t-1])) / tau
        dgrsa = -g_rsa[t-1]/tau_rsa
        dgr = -g_r[t-1]/tau_r
         
        # Integration Step
        u[t] = u[t-1] + dt * du
        g_rsa[t] = g_rsa[t-1] + dt * dgrsa
        g_r[t] = g_r[t-1] + dt * dgr
            
        # Nonlinear operations
        idc_r = np.isclose(u[t-1], u_t)
        idc_t = (u[t-1] > threshold)
        
        g_rsa[t,idc_r] = g_rsa[t-1,idc_r] + delta_g_rsa
        g_r[t,idc_r] = g_r[t-1,idc_r] + delta_g_r
        u[t,idc_t] = u_t
        u[t,idc_r] = u_r

    if flatten:
        u = u[:,0]
        g_rsa = g_rsa[:,0]
        g_r = g_r[:,0]
    
    return u, (g_rsa,g_r)
