import numpy as np
import torch
from scipy.stats import uniform, norm, truncnorm
from payne_optuna.utils import ensure_tensor


class UniformLogPrior:
    def __init__(self, label, lower_bound, upper_bound, out_of_bounds_val=-1e10):
        self.label = label
        self.lower_bound = ensure_tensor(lower_bound)
        self.upper_bound = ensure_tensor(upper_bound)
        self.out_of_bounds_val = ensure_tensor(out_of_bounds_val)
        self.dist = uniform(loc=self.lower_bound, scale=self.upper_bound-self.lower_bound)

    def __call__(self, x):
        log_prior = torch.zeros_like(x)
        log_prior[(x < self.lower_bound) | (x > self.upper_bound)] = self.out_of_bounds_val
        return log_prior

    def sample(self, n_samples):
        return self.dist.rvs(size=n_samples)


class GaussianLogPrior:
    def __init__(self, label, mu, sigma):
        self.label = label
        self.mu = ensure_tensor(mu)
        self.sigma = ensure_tensor(sigma)
        self.lower_bound = ensure_tensor(-np.inf)
        self.upper_bound = ensure_tensor(np.inf)
        self.dist = norm(loc=self.mu, scale=self.sigma)

    def __call__(self, x):
        return torch.log(1.0 / (np.sqrt(2 * np.pi) * self.sigma)) - 0.5 * (x - self.mu) ** 2 / self.sigma ** 2

    def sample(self, n_samples):
        return self.dist.rvs(size=n_samples)


class FlatLogPrior:
    def __init__(self, label):
        self.label = label
        self.lower_bound = ensure_tensor(-np.inf)
        self.upper_bound = ensure_tensor(np.inf)

    def __call__(self, x):
        return torch.zeros_like(x)

    def sample(self, n_samples):
        raise NotImplementedError


class DeltaLogPrior:
    def __init__(self, label, d, tolerance=1e-4, out_of_bounds_val=-1e10):
        self.label = label
        self.d = ensure_tensor(d)
        self.tolerance = ensure_tensor(tolerance)
        self.lower_bound = self.d - self.tolerance
        self.upper_bound = self.d + self.tolerance
        self.out_of_bounds_val = ensure_tensor(out_of_bounds_val)
        self.dist = None

    def __call__(self, x):
        log_prior = torch.zeros_like(x)
        log_prior[(x < self.lower_bound) | (x > self.upper_bound)] = self.out_of_bounds_val
        return log_prior

    def sample(self, n_samples):
        return self.d * torch.ones(n_samples)