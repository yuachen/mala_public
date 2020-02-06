import numpy as np
import scipy.stats

import mcmc
import sys
import time as time


# logconcave function
def f(x, mean = 0., sigma = 1.):
    # x is of dimension n * d
    return 0.5*np.sum((x - mean)**2/sigma**2, axis = 1)

# density of f up to a constant
def density_f(x, mean = 0., sigma = 1.):
    return  np.exp(-f(x, mean, sigma))

def grad_f(x, mean = 0., sigma = 1.):
    # x is of dimension n * d
    return (x - mean)/sigma**2

# logconcave function modified to strongly convex
def f_mod(x, mean = 0., sigma = 1., gamma=1.0):
    # x is of dimension n * d
    return f(x, mean, sigma) + gamma/2.*np.sum((x - mean)**2, axis = 1)

def density_f_mod(x, mean = 0., sigma = 1., gamma=1.0):
    return  np.exp(-f_mod(x, mean, sigma, gamma))

def grad_f_mod(x, mean = 0., sigma = 1., gamma=1.0):
    # x is of dimension n * d
    return grad_f(x, mean, sigma) + gamma * (x - mean)


def main_simu(d, nb_exps=10000, nb_iters=40000, sigma_max=2.0, eps=0.1, seed=1):
    np.random.seed(123456+seed)
    error_ula_all = np.zeros((nb_iters, 1))
    error_ula_02_all = np.zeros((nb_iters, 1))
    error_mala_all = np.zeros((nb_iters, 1))
    error_rwmh_all = np.zeros((nb_iters, 1))

    mean = np.zeros(d)
    sigma =  np.array([1.0 + (sigma_max - 1.0)/(d-1)*i for i in range(d)])
    # make the last sigma large, so that it is close to nonstrongly convex
    sigma[-1] = 1000.
    L = 1./sigma[0]**2
    m = 1./sigma[-1]**2
    kappa = L/m

    print("d = %d, m = %0.2f, L = %0.2f, kappa = %0.2f" %(d, m, L, kappa))

    # modify the objective function to make it gamma/2 strongly convex
    # fourth moment bound
    nu = 1.0
    gamma = 2. * eps/d/nu
    L_mod = L + gamma/2.
    m_mod = m + gamma/2.
    kappa_mod = L_mod/m_mod

    print("d = %d, m_mod = %0.2f, L_mod = %0.2f, kappa_mod = %0.2f, eps = %0.2f" %(d, m_mod, L_mod, kappa_mod, eps))

    # compare error on the before-last dimension
    def error_quantile(x_curr):
        q3 =  sigma[-2]*scipy.stats.norm.ppf(0.75)
        e1 = np.abs(np.percentile(x_curr[:, -2], 75) - q3)/q3
        return np.array([e1])

    init_distr = 1./np.sqrt(L_mod)*np.random.randn(nb_exps, d)

    def grad_f_local(x):
        return grad_f_mod(x, mean=mean, sigma=sigma, gamma=gamma)

    def f_local(x):
        return density_f_mod(x, mean=mean, sigma=sigma, gamma=gamma)

    error_ula_all, x_ula = mcmc.ula(init_distr, grad_f_local, error_quantile,
                                    epsilon=eps*10., kappa=kappa_mod, L=L_mod, nb_iters=nb_iters, nb_exps=nb_exps)
    error_ula_02_all, x_ula_02 = mcmc.ula(init_distr, grad_f_local, error_quantile,
                                          epsilon=eps*1., kappa=kappa_mod, L=L_mod, nb_iters=nb_iters, nb_exps=nb_exps)
    error_mala_all, x_mala = mcmc.mala(init_distr, grad_f_local, f_local, error_quantile,
                                       kappa=kappa_mod, L=L_mod, nb_iters=nb_iters, nb_exps=nb_exps)
    error_rwmh_all, x_rwmh = mcmc.rwmh(init_distr, f_local, error_quantile,
                                       kappa=kappa_mod, L=L_mod, nb_iters=nb_iters, nb_exps=nb_exps)

    result = {}
    result['d'] = d
    result['nb_iters'] = nb_iters
    result['nb_exps'] = nb_exps
    result['sigma_max'] = sigma_max
    result['eps'] = eps
    result['ula'] = error_ula_all
    result['ula_02'] = error_ula_02_all
    result['mala'] = error_mala_all
    result['rwmh'] = error_rwmh_all

    save_path = "PLEASE UPDATE BEFORE USE"
    np.save('%s/gaussian_nonstrongly_nonisotropic_d%d_eps%d_iters%d_exps%d_seed%d.npy' %(save_path, d, int(1./eps), nb_iters, nb_exps, seed), result)

if __name__ == '__main__':
    d = int(sys.argv[1])
    seed = int(sys.argv[2])
    main_simu(d=d, nb_exps=10000, nb_iters=40000, sigma_max=2.0, eps=0.1, seed=seed)


