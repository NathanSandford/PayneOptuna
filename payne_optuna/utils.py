import numpy as np
import torch

def ensure_tensor(input_, precision=torch.float32):
    if isinstance(input_, torch.Tensor):
        return input_.to(precision)
    elif isinstance(input_, np.ndarray):
        return torch.from_numpy(input_).to(precision)
    elif isinstance(input_, (int, float)):
        return torch.Tensor([input_]).to(precision)
    elif isinstance(input_, list):
        return torch.Tensor(input_).to(precision)
    else:
        raise TypeError(f"input_ type ({type(input_)}) cannot be converted to a Tensor")


def j_nu(x, nu, n_tau=100):
    x_ = x.view(-1, 1)
    tau = torch.linspace(0, np.pi, n_tau).view(1,-1)
    integrand = torch.cos(nu*tau - x_ * torch.sin(tau))
    return (1/np.pi) * torch.trapz(integrand, tau, dim=-1)


def interp(x, y, x_new, fill=None):
    y_ = y.unsqueeze(0) if y.ndim == 1 else y
    out_of_bounds = (x_new < x[0]) | (x_new > x[-1])
    if fill is None and torch.any(out_of_bounds):
        raise ValueError("A value in x_new is outside of the interpolation range.")
    x_new_indices = torch.searchsorted(x, x_new)
    x_new_indices = x_new_indices.clamp(1, x.shape[0] - 1)
    lo = x_new_indices - 1
    hi = x_new_indices
    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y_[:, lo]
    y_hi = y_[:, hi]
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    y_new = slope * (x_new - x_lo) + y_lo
    y_new[:, out_of_bounds] = fill
    return y_new


def log_lambda_grid(dv, min_wave, max_wave):
    max_log_dv = np.log10(dv / 2.99792458e5 + 1.0)
    log_min_wave = np.log10(min_wave)
    log_max_wave = np.log10(max_wave)
    min_n_pixels = (log_max_wave - log_min_wave) / max_log_dv
    n_pixels = 2
    while n_pixels < min_n_pixels:
        n_pixels *= 2
    log_dv = (log_max_wave - log_min_wave) / (n_pixels - 1)
    wave = 10 ** (log_min_wave + log_dv * np.arange(n_pixels))
    return wave
