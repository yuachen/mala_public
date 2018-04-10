# This is code to support simulations in the paper 
#"Log-concave sampling: Metropolis-Hastings algorithms are fast!"
# by Raaz Dwivedi, Yuansi Chen, Martin J. Wainwright, Bin Yu
# mcmc package that contains random walk with Metripolis-Hastings,
# Unadjusted Langevin Dynamics and Metropolis Adjusted Langevin Dynamics
import numpy as np


def density_normal(x, mean=0., sigma=1.):
    """
    Density of the normal distribution up to constant
    Args:
        x (n * d): location, can be high dimension
        mean (d): mean
    Returns:
        density value at x
    """
    return np.exp(-np.sum((x - mean)**2, axis=1) / 2 / sigma**2)


def ula(x_init, grad_f, error_metric, epsilon=1.0, kappa=1.0, L=1.0,
        nb_iters=200, nb_exps=10):
    """
    Unadjusted Langevin Dynamics for sampling from a logconcave distribution.

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        grad_f (fun):  compute the gradient of the target f function,
            taking current distribution as argument and returning the gradient
        error_metric (fun): compute the error metric
        epsilon (float): the error threshold for ULA,
            used to determine stepsize
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
    """
    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_ula = 0.5 * epsilon**2 / d / kappa / L
    nh_ula = np.sqrt(2 * h_ula)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1

    for i in range(nb_iters - 1):
        x_curr = x_curr - h_ula * grad_f(x_curr) \
            + nh_ula * np.random.randn(nb_exps, d)

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr


def mala(x_init, grad_f, f, error_metric, kappa=1.0, L=1.0, nb_iters=200, nb_exps=10):
    """
    Metropolis Adjusted Langevin Dynamics
    for sampling from a logconcave distribution.

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        grad_f (fun):  compute the gradient of the target f function,
            taking current distribution as argument and returning the gradient
        f (fun):  compute the density of the target f function up to constant,
        error_metric (fun): compute the error metric
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
    """

    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_mala = 0.5 / L / np.maximum(d, np.sqrt(d * kappa))
    nh_mala = np.sqrt(2 * h_mala)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1

    for i in range(nb_iters - 1):
        proposal = x_curr - h_mala * grad_f(x_curr) \
            + nh_mala * np.random.randn(nb_exps, d)

        ratio = f(proposal) \
            * density_normal(x=x_curr,
                             mean=proposal - h_mala * grad_f(proposal),
                             sigma=nh_mala)
        ratio /= f(x_curr) \
            * density_normal(x=proposal,
                             mean=x_curr - h_mala * grad_f(x_curr),
                             sigma=nh_mala)

        # Metropolis Hastings step
        ratio = np.minimum(1., ratio)
        a = np.random.rand(nb_exps)
        index_forward = np.where(a <= ratio)[0]

        x_curr[index_forward, ] = proposal[index_forward, ]

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr


def rwmh(x_init, f, error_metric, kappa=1.0, L=1.0, nb_iters=200, nb_exps=10):
    """
    Random walk with Metropolis Hastings
    for sampling from a logconcave distribution.

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        f (fun):  compute the density of the target f function up to constant,
        error_metric (fun): compute the error metric
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
    """
    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_rwmh = 0.5 / d**2 / kappa / L
    nh_rwmh = np.sqrt(2 * h_rwmh)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1

    for i in range(nb_iters - 1):
        proposal = x_curr + nh_rwmh * np.random.randn(nb_exps, d)

        ratio = f(proposal) \
            * density_normal(x=x_curr,
                             mean=proposal,
                             sigma=nh_rwmh)
        ratio /= f(x_curr) \
            * density_normal(x=proposal,
                             mean=x_curr,
                             sigma=nh_rwmh)

        ratio = np.minimum(1., ratio)
        a = np.random.rand(nb_exps)
        index_forward = np.where(a <= ratio)[0]

        x_curr[index_forward, ] = proposal[index_forward, ]

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr
