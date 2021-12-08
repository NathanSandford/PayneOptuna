import numpy as np
import torch

from payne_optuna.fitting import UniformLogPrior
from payne_optuna.utils import ensure_tensor

import emcee

def clamp_p0(p0_, label_names, priors, model):
    fe_idx = model.labels.index('Fe')
    stellar_labels_unscaled_ = model.unscale_stellar_labels(ensure_tensor(p0_)[:, :model.n_stellar_labels])
    other_labels_ = ensure_tensor(p0_)[:, model.n_stellar_labels:]
    stellar_labels_unscaled = torch.zeros_like(stellar_labels_unscaled_)
    other_labels = torch.zeros_like(other_labels_)
    # Clamp Fe first
    fe_prior = priors['stellar_labels'][fe_idx]
    fe_lower_bound = fe_prior.lower_bound.item() if type(fe_prior) == UniformLogPrior else -np.inf
    fe_upper_bound = fe_prior.upper_bound.item() if type(fe_prior) == UniformLogPrior else np.inf
    stellar_labels_unscaled[:, fe_idx] = stellar_labels_unscaled_[:, fe_idx].clamp(
        min=fe_lower_bound+1e-4,
        max=fe_upper_bound-1e-4,
    )
    # Clamp remaining stellar labels
    for label in model.labels:
        idx = model.labels.index(label)
        prior = priors['stellar_labels'][idx]
        lower_bound = prior.lower_bound.item() if type(prior) == UniformLogPrior else -np.inf
        upper_bound = prior.upper_bound.item() if type(prior) == UniformLogPrior else np.inf
        if label == 'Fe':
            continue
        if label in ['Teff', 'logg', 'v_micro',]:
            stellar_labels_unscaled[:, idx] = stellar_labels_unscaled_[:, idx].clamp(
                min=lower_bound+1e-4,
                max=upper_bound-1e-4,
            )
        else:
            stellar_labels_unscaled[:, idx] = (stellar_labels_unscaled_[:, idx] - stellar_labels_unscaled[:, fe_idx]).clamp(
                min=lower_bound+1e-4,
                max=upper_bound-1e-4,
            ) + stellar_labels_unscaled[:, fe_idx]
    # Clamp additional labels
    for i, label in enumerate(label_names[model.n_stellar_labels:]):
        prior = priors[label]
        lower_bound = prior.lower_bound.item() if type(prior) == UniformLogPrior else -np.inf
        upper_bound = prior.upper_bound.item() if type(prior) == UniformLogPrior else np.inf
        other_labels[:, i] = other_labels_[:, i].clamp(min=lower_bound, max=upper_bound)
    p0 = torch.hstack(
        [model.scale_stellar_labels(stellar_labels_unscaled), other_labels]
    ).detach().numpy()
    return p0


class MCMCReader:
    def __init__(self, mcmc_file, emulator, label_names, use_gaia_phot=True, load_burn_in=False):
        self.mcmc_file = mcmc_file
        self.emulator = emulator
        self.label_names = label_names
        self.use_gaia_phot = use_gaia_phot
        self.load_burn_in = load_burn_in
        if self.load_burn_in:
            self.burn_in_sampler = emcee.backends.HDFBackend(
                mcmc_file,
                "burn_in_1",
                read_only=True
            )
            self.burn_in_nwalkers = self.burn_in_sampler.shape[0]
            self.burn_in_nlabels = self.burn_in_sampler.shape[1]
            self.burn_in_nsteps = self.burn_in_sampler.iteration
        self.sampler = emcee.backends.HDFBackend(
            mcmc_file,
            "samples",
            read_only=True
        )
        self.tau = int(np.max(self.sampler.get_autocorr_time(tol=0)))
        self.nwalkers = self.sampler.shape[0]
        self.nlabels = self.sampler.shape[1]
        self.nsteps = self.sampler.iteration

        if self.load_burn_in:
            self.burn_in_scaled_chains, self.burn_in_unscaled_chains \
                = self.get_chains(self.burn_in_sampler)
        self.scaled_chains, self.unscaled_chains = self.get_chains(self.sampler)
        self.scaled_samples, self.unscaled_samples = self.get_samples()

        self.scaled_mean = self.scaled_samples.mean(axis=0)
        self.scaled_std = self.scaled_samples.std(axis=0)
        self.unscaled_mean = self.unscaled_samples.mean(axis=0)
        self.unscaled_std = self.unscaled_samples.std(axis=0)

    def get_chains(self, sampler):
        if self.use_gaia_phot:
            scaled_chains = np.concatenate([
                np.zeros((self.nsteps, self.nwalkers, 1)),
                np.zeros((self.nsteps, self.nwalkers, 1)),
                sampler.get_chain(),
            ], axis=2)
        else:
            scaled_chains = sampler.get_chain()
        unscaled_stellar_chains = self.emulator.unscale_stellar_labels(
            ensure_tensor(
                scaled_chains[:, :, :self.emulator.n_stellar_labels].reshape(-1, self.emulator.n_stellar_labels)
            )
        ).reshape(-1, self.nwalkers, self.emulator.n_stellar_labels).detach().numpy()
        unscaled_chains = np.concatenate([
            unscaled_stellar_chains,
            scaled_chains[:, :, self.emulator.n_stellar_labels:]
        ], axis=-1)
        for i, label in enumerate(self.label_names):
            if label not in ['Teff', 'logg', 'v_micro', 'Fe', 'inst_res', 'log_vmacro', 'log_vsini', 'rv', ]:
                unscaled_chains[:, :, i] -= unscaled_chains[:, :, self.emulator.labels.index('Fe')]
            elif label in ['log_vmacro', 'log_vsini']:
                unscaled_chains[:, :, i] = 10 ** unscaled_chains[:, :, i]
            elif label in ['rv']:
                unscaled_chains[:, :, i] = self.emulator.rv_scale * unscaled_chains[:, :, i]
        return scaled_chains, unscaled_chains

    def get_samples(self):
        _scaled_samples = self.sampler.get_chain(
            discard=5 * self.tau,
            thin=int(np.ceil(self.tau / 2)),
            flat=True,
        )
        self.nsamples = _scaled_samples.shape[0]
        if self.use_gaia_phot:
            scaled_samples = np.concatenate([
                np.zeros((self.nsamples, 1)),
                np.zeros((self.nsamples, 1)),
                _scaled_samples,
            ], axis=1)
        else:
            scaled_samples = _scaled_samples
        unscaled_stellar_samples = self.emulator.unscale_stellar_labels(
            ensure_tensor(scaled_samples[:, :self.emulator.n_stellar_labels])
        ).detach().numpy()
        unscaled_samples = np.concatenate([
            unscaled_stellar_samples,
            scaled_samples[:, self.emulator.n_stellar_labels:]
        ], axis=1)
        for i, label in enumerate(self.label_names):
            if label not in ['Teff', 'logg', 'v_micro', 'Fe', 'inst_res', 'log_vmacro', 'log_vsini', 'rv', ]:
                unscaled_samples[:, i] -= unscaled_samples[:, self.emulator.labels.index('Fe')]
            elif label in ['log_vmacro', 'log_vsini']:
                unscaled_samples[:, i] = 10 ** unscaled_samples[:, i]
            elif label in ['rv']:
                unscaled_samples[:, i] = self.emulator.rv_scale * unscaled_samples[:, i]
        return scaled_samples, unscaled_samples
