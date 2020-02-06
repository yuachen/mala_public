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


def main_simu(nb_exps=10000, nb_iters=40000, sigma_max=2.0, seed=1):
    d = 2
    epsilons = np.arange(10)*0.1+0.1
    np.random.seed(123456+seed)
    error_ula_all = np.zeros((epsilons.shape[0], nb_iters, 1))
    error_mala_all = np.zeros((nb_iters, 1))
    error_rwmh_all = np.zeros((nb_iters, 1))

    mean = np.zeros(d)
    sigma =  np.array([1.0 + (sigma_max - 1.0)/(d-1)*i for i in range(d)])
    L = 1./sigma[0]**2
    m = 1./sigma[-1]**2
    kappa = L/m

    print("d = %d, m = %0.2f, L = %0.2f, kappa = %0.2f" %(d, m, L, kappa))

    def error_quantile(x_curr):
        q3 =  sigma[-1]*scipy.stats.norm.ppf(0.75)
        e1 = np.abs(np.percentile(x_curr[:, -1], 75) - q3)/q3
        return np.array([e1])

    init_distr = 1./np.sqrt(L)*np.random.randn(nb_exps, d)

    def grad_f_local(x):
        return grad_f(x, mean=mean, sigma=sigma)

    def f_local(x):
        return density_f(x, mean=mean, sigma=sigma)

    error_mala_all, x_mala = mcmc.mala(init_distr, grad_f_local, f_local, error_quantile,
                                       kappa=kappa, L=L, nb_iters=nb_iters, nb_exps=nb_exps)
    error_rwmh_all, x_rwmh = mcmc.rwmh(init_distr, f_local, error_quantile,
                                       kappa=kappa, L=L, nb_iters=nb_iters, nb_exps=nb_exps)

    for j, eps in enumerate(epsilons):
        error_ula_all[j], x_ula = mcmc.ula(init_distr, grad_f_local, error_quantile,
                                    epsilon=eps, kappa=kappa, L=L, nb_iters=nb_iters, nb_exps=nb_exps)

    result = {}
    result['epsilons'] = epsilons
    result['d'] = d
    result['nb_iters'] = nb_iters
    result['nb_exps'] = nb_exps
    result['sigma_max'] = sigma_max
    result['ula'] = error_ula_all
    result['mala'] = error_mala_all
    result['rwmh'] = error_rwmh_all

    save_path = "PLEASE UPDATE BEFORE USE"
    np.save('%s/gaussian_nonisotropic_eps_iters%d_exps%d_seed%d.npy' %(save_path, nb_iters, nb_exps, seed), result)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    main_simu(nb_exps=10000, nb_iters=40000, sigma_max=2.0, seed=seed)


