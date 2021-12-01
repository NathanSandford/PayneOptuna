import argparse
import yaml
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from payne_optuna.fitting import PayneStitchedEmulator
from payne_optuna.fitting import UniformLogPrior, GaussianLogPrior, FlatLogPrior, gaussian_log_likelihood
from payne_optuna.utils import ensure_tensor, find_runs, noise_up_spec
from payne_optuna.misc import model_io

import emcee
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from corner import corner


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Fit Observed Spectrum w/ Optimizer")
    parser.add_argument("fitting_configs", help="Config File describing the fitting procedure")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


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
    for i, label in enumerate(set(model.labels) ^ set(label_names)):
        prior = priors[label]
        lower_bound = prior.lower_bound.item() if type(prior) == UniformLogPrior else -np.inf
        upper_bound = prior.upper_bound.item() if type(prior) == UniformLogPrior else np.inf
        other_labels[:, i] = other_labels_[:, i].clamp(min=lower_bound, max=upper_bound)
    p0 = torch.hstack(
        [model.scale_stellar_labels(stellar_labels_unscaled), other_labels]
    ).detach().numpy()
    return p0


def main(args):
    ##################################
    ######## PLOTTING CONFIGS ########
    ##################################
    mpl.rc('axes', grid=True, lw=2)
    mpl.rc('ytick', direction='in', labelsize=10)
    mpl.rc('ytick.major', size=5, width=1)
    mpl.rc('xtick', direction='in', labelsize=10)
    mpl.rc('xtick.major', size=5, width=1)
    mpl.rc('ytick', direction='in', labelsize=10)
    mpl.rc('ytick.major', size=5, width=1)
    mpl.rc('grid', alpha=0.75, lw=1)
    mpl.rc('legend', edgecolor='k', framealpha=1, fancybox=False)
    mpl.rc('figure', dpi=300)

    #####################
    ######## I/O ########
    #####################
    with open(args.fitting_configs) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    # Parse Observation
    program = configs['observation']['program']
    date = configs['observation']['date']
    star = configs['observation']['star']
    frame = configs['observation']['frame']
    orders = configs['observation']['orders']
    resolution = configs['observation']['resolution']
    default_res = configs['observation']['default_res']
    model_res = configs['observation']['model_res']
    bin_errors = configs['observation']['bin_errors']
    snr_rdx = configs['observation']['snr_rdx']
    snr_tag = f'_snr{snr_rdx:02.0f}' if snr_rdx is not False else ''
    obs_name = f'{star}_{frame}_{date}'
    # I/O Prep
    model_config_dir = Path(configs['paths']['model_config_dir'])
    data_dir = Path(configs['paths']['data_dir'])
    program_dir = data_dir.joinpath(f'{program}')
    mask_dir = program_dir.joinpath('masks')
    fig_dir = program_dir.joinpath(f'figures/{star}_{date}')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = program_dir.joinpath(f'fits/{star}_{date}')
    fits_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = data_dir.joinpath('samples')
    sample_dir.mkdir(parents=True, exist_ok=True)
    model_config_files = sorted(list(model_config_dir.glob('*')))
    line_mask_file = mask_dir.joinpath(f"{configs['masks']['line']}")
    nlte_errs_files = sorted(list(mask_dir.joinpath(configs['masks']['nlte']).glob('*'))) \
        if configs['masks']['nlte'] is not False else False

    ###############################
    ######## LOAD BEST FIT ########
    ###############################
    # Find Best Fit from Optimizer
    fit_files = sorted(list(fits_dir.glob(f"{obs_name}_fit_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_*.npz")))
    if len(fit_files) == 0:
        raise RuntimeError(
            f"Could not load optimizer solutions."
        )
    losses = np.zeros(len(fit_files))
    for i, fit_file in enumerate(fit_files):
        tmp = np.load(fit_file)
        losses[i] = tmp['loss']
    best_idx = np.argmin(losses)
    print(f'Best from optimization: Trial {best_idx + 1}')
    optim_fit = np.load(fit_files[i])
    # Load Observation
    obs = {}
    obs['wave'] = optim_fit['obs_wave']
    obs['flux'] = optim_fit['obs_flux']
    obs['errs'] = optim_fit['obs_errs']
    obs['blaz'] = optim_fit['obs_blaz']
    obs['mask'] = optim_fit['obs_mask']
    # Load Photometry & CMD Model
    if configs['fitting']['use_gaia_phot']:
        phot = pd.read_hdf(data_dir.joinpath('photometry.h5'), 'M15')
        obs['bp-rp'] = phot.loc[star, 'bp-rp']
        obs['g'] = phot.loc[star, 'g']
        gaia_cmd_interp = np.load(data_dir.joinpath('gaia_cmd_interp.npy'), allow_pickle=True)[()]

    ########################
    ######## MODELS ########
    ########################
    # Load Line Masks
    print(f'Loading line masks')
    if line_mask_file is not False:
        with open(line_mask_file) as file:
            line_masks = yaml.load(file, Loader=yaml.FullLoader)
    else:
        line_masks = []
    # Load NLTE Uncertainties
    if nlte_errs_files is not False:
        print('Loading NLTE Uncertainties')
        nlte_errs = []
        for file in nlte_errs_files:
            tmp = np.load(file)
            nlte_errs.append(tmp['errs'])
    else:
        nlte_errs = np.zeros(len(model_config_files))
    # Load Models
    models = []
    for i, model_config_file in enumerate(model_config_files):
        model = model_io.load_model(model_config_file)
        #  Mask Lines
        model_io.mask_lines(model, line_masks, mask_value=1.0)
        # Mask NLTE Lines
        model.mod_errs = np.sqrt(model.mod_errs ** 2 + nlte_errs[i] ** 2)
        models.append(model)
    # Sort Models by Ascending Wavelength
    models = [models[i] for i in np.argsort([model.wavelength.min() for model in models])]
    # Initialize Emulator
    payne = PayneStitchedEmulator(
        models=models,
        cont_deg=configs['fitting']['cont_deg'],
        cont_wave_norm_range=(-10, 10),
        obs_wave=obs['wave'],
        obs_blaz=obs['blaz'],
        include_model_errs=True,
        model_res=model_res,
        vmacro_method='iso_fft',
    )

    ######################################
    ######## DEGRADE SIGNAL/NOISE ########
    ######################################
    if snr_rdx is not False:
        print(f'Increase Noise by a Factor of {snr_rdx}')
        D, sigma_D = noise_up_spec(obs['flux'], obs['errs'], snr_rdx, seed=8645)
        obs['flux'] = D
        obs['errs'] = sigma_D

    ###############################
    ######## PLOT SPECTRUM ########
    ###############################
    # Plot Observed Spectrum & Blaze Function
    if configs['output']['plot_obs']:
        print('Plotting 1D spectrum and blaze function')
        n_ord = obs['wave'].shape[0]
        fig = plt.figure(figsize=(10, n_ord))
        gs = GridSpec(n_ord, 1)
        gs.update(hspace=0.5)
        for j in range(n_ord):
            ax = plt.subplot(gs[j, 0])
            ax.plot(obs['wave'][j], obs['blaz'][j], alpha=0.8, c='r', label='Scaled Blaze')
            ax.scatter(
                obs['wave'][j],
                obs['flux'][j],
                alpha=0.8, marker='.', s=1, c='k', label='Observed Spectrum'
            )
            for k, conv_mask_range in enumerate(find_runs(0.0, obs['mask'][j])):
                label = 'Mask' if k == 0 else ''
                ax.axvspan(
                    obs['wave'][j, conv_mask_range[0]],
                    obs['wave'][j, np.min([len(obs['wave'][j])-1, conv_mask_range[1]])],
                    color='grey', alpha=0.2, label=label,
                )
            ax.set_ylim(0, 1.5 * np.quantile(obs['flux'][j][obs['mask'][j]], 0.95))
            if j == 0:
                ax.set_title(obs_name)
                ax.legend(fontsize=8)
        plt.savefig(fig_dir.joinpath(f'{obs_name}_obs_{resolution}{snr_tag}.mcmc.png'))
        plt.close('all')

    ####################################
    ######## PRIORS + POSTERIOR ########
    ####################################
    # Set Priors
    print('Setting Priors')
    stellar_label_priors = []
    for i, label in enumerate(payne.labels):
        if label in configs['fitting']['priors']:
            if (label in ['Teff', 'logg']) and configs['fitting']['use_gaia_phot']:
                stellar_label_priors.append(
                    UniformLogPrior(
                        label,
                        payne.unscale_stellar_labels(-0.55 * torch.ones(payne.n_stellar_labels))[i],
                        payne.unscale_stellar_labels(0.55 * torch.ones(payne.n_stellar_labels))[i],
                    )
                )
            elif configs['fitting']['priors'][label][0] == 'N':
                stellar_label_priors.append(
                    GaussianLogPrior(
                        label,
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2]
                    )
                )
            elif configs['fitting']['priors'][label][0] == 'U':
                stellar_label_priors.append(
                    UniformLogPrior(
                        label,
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2],
                        out_of_bounds_val=-np.inf,
                    )
                )
            else:
                raise KeyError(f"Cannot parse prior info for {label}")
        else:
            stellar_label_priors.append(
                UniformLogPrior(
                    label,
                    payne.unscale_stellar_labels(-0.55 * torch.ones(payne.n_stellar_labels))[i],
                    payne.unscale_stellar_labels(0.55 * torch.ones(payne.n_stellar_labels))[i],
                    out_of_bounds_val=-np.inf,
                )
            )
    priors = {
        "stellar_labels": stellar_label_priors,
        'log_vmacro': UniformLogPrior('log_vmacro', -1, 1.3, -np.inf),
        'log_vsini': FlatLogPrior('log_vsini'),
        'inst_res': FlatLogPrior('inst_res') if resolution == 'default' else
        GaussianLogPrior('inst_res', int(resolution), 0.01 * int(resolution)),
        'rv': FlatLogPrior('rv'),
    }
    # Define Posterior Function
    def log_probability(theta, model, obs, priors, cmd_interp_fn=None):
        fe_idx = model.labels.index('Fe')
        nwalkers = theta.shape[0]
        if cmd_interp_fn:
            stellar_labels = np.hstack([
                np.zeros((nwalkers,2)),
                theta[:, :model.n_stellar_labels-2]
            ])
            unscaled_stellar_labels = model.unscale_stellar_labels(ensure_tensor(stellar_labels)).detach().numpy()
            fe_phot = np.vstack([
                unscaled_stellar_labels[:, fe_idx],
                obs['bp-rp']*np.ones(nwalkers),
                obs['g']*np.ones(nwalkers)
            ]).T
            logg, logTeff = cmd_interp_fn(fe_phot).T
            unscaled_stellar_labels[:, 0] = 10**logTeff
            unscaled_stellar_labels[:, 1] = logg
            unscaled_stellar_labels = ensure_tensor(unscaled_stellar_labels)
            stellar_labels = model.scale_stellar_labels(ensure_tensor(unscaled_stellar_labels))
        else:
            stellar_labels = theta[:, :model.n_stellar_labels]
            unscaled_stellar_labels = model.unscale_stellar_labels(ensure_tensor(stellar_labels))
        reverse_idx = -1
        rv = theta[:, reverse_idx]
        if configs['fitting']['fit_vmacro']:
            reverse_idx -= 1
            log_vmacro = theta[:, reverse_idx]
        else:
            log_vmacro = None
        if configs['fitting']['fit_vsini']:
            reverse_idx -= 1
            log_vsini = theta[:, reverse_idx]
        else:
            log_vsini = None
        if configs['fitting']['fit_inst_res']:
            reverse_idx -= 1
            inst_res = theta[:, reverse_idx]
        else:
            inst_res = None
        cont_coeffs = optim_fit["cont_coeffs"]
        mod_spec, mod_errs = model(
            stellar_labels=ensure_tensor(stellar_labels),
            rv=ensure_tensor(rv),
            vmacro=ensure_tensor(10 ** log_vmacro) if log_vmacro is not None else None,
            vsini=ensure_tensor(10 ** log_vsini) if log_vsini is not None else None,
            inst_res=ensure_tensor(inst_res) if inst_res is not None else None,
            cont_coeffs=ensure_tensor(cont_coeffs),
        )
        log_likelihood = gaussian_log_likelihood(
            pred=mod_spec,
            target=ensure_tensor(obs['flux']),
            pred_errs=mod_errs,
            target_errs=ensure_tensor(obs['errs']),
        )
        log_priors = torch.zeros_like(log_likelihood)
        for i, label in enumerate(model.labels):
            if label in ['Teff', 'logg', 'v_micro', 'Fe']:
                log_priors += priors['stellar_labels'][i](unscaled_stellar_labels[:, i])
            else:
                log_priors += priors['stellar_labels'][i](
                    unscaled_stellar_labels[:, i] - unscaled_stellar_labels[:, fe_idx]
                )
        if log_vmacro is not None:
            log_priors += priors['log_vmacro'](ensure_tensor(log_vmacro))
        if log_vsini is not None:
            log_priors += priors['log_vsini'](ensure_tensor(log_vsini))
        if inst_res is not None:
            log_priors += priors['inst_res'](ensure_tensor(inst_res))
        log_priors += priors['rv'](ensure_tensor(rv))
        return (log_likelihood + log_priors).detach().numpy()

    #################################
    ######## BURN IN WALKERS ########
    #################################
    ### Run Burn-In 1 ###
    # Initialize Walkers
    p0_list = [optim_fit["stellar_labels"][0]]
    label_names = deepcopy(payne.labels)
    if configs['fitting']['fit_inst_res']:
        p0_list.append(optim_fit["inst_res"][0])
        label_names.append("inst_res")
    if configs['fitting']['fit_vsini']:
        p0_list.append(optim_fit["log_vsini"][0])
        label_names.append("log_vsini")
    if configs['fitting']['fit_vmacro']:
        p0_list.append(optim_fit["log_vmacro"][0])
        label_names.append("log_vmacro")
    p0_list.append(optim_fit["rv"])
    label_names.append("rv")
    p0_ = np.concatenate(p0_list) + 0.1 * np.random.randn(128, len(label_names))
    p0 = clamp_p0(p0_, label_names, priors, payne)
    if configs['fitting']['use_gaia_phot']:
        p0 = p0[:, 2:]
    nwalkers, ndim = p0.shape
    # Initialize Backend
    sample_file = sample_dir.joinpath(f"{obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.h5")
    backend = emcee.backends.HDFBackend(sample_file, name=f"burn_in_1")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
        args=(
            payne,
            obs,
            priors,
            gaia_cmd_interp if configs['fitting']['use_gaia_phot'] else None
        ),
        vectorize=True,
        backend=backend,
    )
    if not np.all(np.isfinite(sampler.compute_log_prob(p0)[0])):
        raise RuntimeError("Walkers are improperly initialized")
    # Run Sampler until walkers stop wandering
    p_mean_last = p0.mean(0)
    log_prob_mean_last = -np.inf
    converged = False
    for _ in sampler.sample(p0, iterations=10000, progress=True, store=True):
        if (sampler.iteration % 100):
            continue
        p_mean = sampler.get_chain(flat=True, thin=1, discard=sampler.iteration - 100).mean(0)
        log_prob_mean = sampler.get_log_prob(discard=sampler.iteration - 1).mean()
        print(f" ({obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag})")
        print(f'max(dMean) = {np.max(p_mean - p_mean_last):0.04f}')
        print(f'mean log(d_logP) = {np.log10(np.abs((log_prob_mean - log_prob_mean_last) / log_prob_mean)):0.2f}')
        if np.abs(np.max(p_mean - p_mean_last)) < 0.001 \
                and (sampler.iteration >= 1000) \
                and np.abs(log_prob_mean - log_prob_mean_last) / log_prob_mean <= 1e-5:
            p_mean_last = p_mean
            log_prob_mean_last = log_prob_mean
            converged = True
            break
        p_mean_last = p_mean
        log_prob_mean_last = log_prob_mean
    if configs['output']['plot_chains']:
        samples = sampler.get_chain()
        if configs['fitting']['use_gaia_phot']:
            nlabels = len(label_names) - 2
            _label_names = label_names[2:]
        else:
            nlabels = len(label_names)
            _label_names = label_names
        fig, axes = plt.subplots(nlabels, figsize=(10, 20), sharex=True)
        for i in range(nlabels):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(_label_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(
            fig_dir.joinpath(f"{obs_name}_burnin_1_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.png"))
    print('Burn-In 1 Complete')

    ##############################
    ######## RUN SAMPLING ########
    ##############################
    # Initialize Walkers
    nwalkers = 128
    resume_from_burnin1 = False
    last_state = sampler.get_last_sample()
    if resume_from_burnin1:
        print("Resuming from Burn-In")
        p0 = last_state.coords
    elif configs['fitting']['use_gaia_phot']:
        best_walker = last_state.coords[last_state.log_prob.argmax()]
        p0_ = np.hstack([
            np.zeros((nwalkers, 1)),
            np.zeros((nwalkers, 1)),
            best_walker + np.random.normal(
                loc=0,
                scale=last_state.coords.std(axis=0) / 2,
                size=(nwalkers, best_walker.shape[0])
            )
        ])
        p0 = clamp_p0(p0_, label_names, priors, payne)[:, 2:]
    else:
        best_walker = last_state.coords[last_state.log_prob.argmax()]
        p0_ = best_walker + 1e-3 * np.random.randn(nwalkers, best_walker.shape[0])
        p0 = clamp_p0(p0_, label_names, priors, payne)
    nwalkers, ndim = p0.shape
    # Initialize Backend
    backend = emcee.backends.HDFBackend(sample_file, name=f"samples")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
        args=(
            payne,
            obs,
            priors,
            gaia_cmd_interp if configs['fitting']['use_gaia_phot'] else None
        ),
        vectorize=True,
        backend=backend,
    )
    # Sample Until Convergence is Reached
    max_steps = int(1e5)
    index = 0
    autocorr = np.empty(max_steps)
    old_tau = np.inf
    for _ in sampler.sample(p0, iterations=max_steps, progress=True, store=True):
        if sampler.iteration % 100:
            continue
        # Check Convergence
        tau = sampler.get_autocorr_time(
            discard=int(np.max(old_tau)) if np.all(np.isfinite(old_tau)) else 0,
            tol=0
        )
        autocorr[index] = np.mean(tau)
        print(
            f"{obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag} " +
            f"Step {sampler.iteration}: Tau = {np.max(tau):.0f}, " +
            f"t/20Tau = {sampler.iteration / (20 * np.max(tau)):.2f},\n" +
            f"dTau/Tau = {np.max(np.abs(old_tau - tau) / tau):.3f}, " +
            f"mean acceptance fraction = {sampler.acceptance_fraction.mean():.2f}"
        )
        index += 1
        # Check convergence
        converged = np.all(tau * 20 < sampler.iteration)
        converged &= np.all((tau - old_tau) / tau < 0.01)
        old_tau = tau
        if converged:
            break
        if sampler.iteration % 500:
            continue
        # Plot Chain Update
        if configs['output']['plot_chains']:
            samples = sampler.get_chain()
            if configs['fitting']['use_gaia_phot']:
                nlabels = len(label_names) - 2
                _label_names = label_names[2:]
            else:
                nlabels = len(label_names)
                _label_names = label_names
            fig, axes = plt.subplots(nlabels, figsize=(10, 20), sharex=True)
            for i in range(nlabels):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(_label_names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            plt.savefig(fig_dir.joinpath(f"{obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_ckpt.png"))

    #################################
    ######## PROCESS SAMPLES ########
    #################################
    if configs['fitting']['use_gaia_phot']:
        # Get Chains
        scaled_samples = np.concatenate([
            np.zeros((sampler.iteration, nwalkers, 1)),
            np.zeros((sampler.iteration, nwalkers, 1)),
            sampler.get_chain(),
        ], axis=2)
        # Get Flat Samples
        _scaled_flat_samples = sampler.get_chain(
            discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2), flat=True
        )
        scaled_flat_samples = np.concatenate([
            np.zeros((_scaled_flat_samples.shape[0], 1)),
            np.zeros((_scaled_flat_samples.shape[0], 1)),
            _scaled_flat_samples,
        ], axis=1)
    else:
        # Get Chains
        scaled_samples = sampler.get_chain()
        # Get Flat Samples
        scaled_flat_samples = sampler.get_chain(
            discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2), flat=True
        )
    # Unscale Samples
    unscaled_samples = payne.unscale_stellar_labels(
        ensure_tensor(
            scaled_samples[:, :, :payne.n_stellar_labels].reshape(-1, payne.n_stellar_labels)
        )
    ).reshape(-1, nwalkers, payne.n_stellar_labels).detach().numpy()
    unscaled_flat_samples = payne.unscale_stellar_labels(
        ensure_tensor(scaled_flat_samples[:, :payne.n_stellar_labels])
    ).detach().numpy()
    # Get Teff and log(g) from [Fe/H] and Photometry
    if configs['fitting']['use_gaia_phot']:
        for i in tqdm(range(nwalkers)):
            fe_phot = np.vstack([
                unscaled_samples[:, i, payne.labels.index('Fe')],
                obs['bp-rp'] * np.ones(sampler.iteration),
                obs['g'] * np.ones(sampler.iteration)
            ]).T
            logg, logTeff = gaia_cmd_interp(fe_phot).T
            unscaled_samples[:, i, 0] = 10 ** logTeff
            unscaled_samples[:, i, 1] = logg
        fe_phot_flat = np.vstack([
            unscaled_flat_samples[:, payne.labels.index('Fe')],
            obs['bp-rp'] * np.ones(unscaled_flat_samples.shape[0]),
            obs['g'] * np.ones(unscaled_flat_samples.shape[0])
        ]).T
        logg, logTeff = gaia_cmd_interp(fe_phot_flat).T
        scaled_flat_samples[:, 0] = 10 ** logTeff
        scaled_flat_samples[:, 1] = logg
    samples = np.concatenate([
        unscaled_samples,
        scaled_samples[:, :, payne.n_stellar_labels:]
    ], axis=-1)
    flat_samples = np.concatenate([
        unscaled_flat_samples,
        scaled_flat_samples[:, payne.n_stellar_labels:]
    ], axis=1)
    # Calculate Mean and Standard Deviation
    scaled_mean = scaled_flat_samples.mean(axis=0)
    scaled_std = scaled_flat_samples.std(axis=0)
    unscaled_mean = unscaled_flat_samples.mean(axis=0)
    unscaled_std = unscaled_flat_samples.std(axis=0)
    print(f"{obs_name} Sampling Summary:")
    for i, label in enumerate(label_names):
        if label == "Fe":
            print(
                f'[{label}/H]\t = {unscaled_mean[i]:.4f} ' +
                f'+/- {unscaled_std[i]:.4f} ({scaled_mean[i]:.4f} +/- {scaled_std[i]:.4f})'
            )
        elif label in ["Teff", "logg", "v_micro"]:
            print(
                f'{label}\t = {unscaled_mean[i]:.4f} ' +
                f'+/- {unscaled_std[i]:.4f} ({scaled_mean[i]:.4f} +/- {scaled_std[i]:.4f})'
            )
        elif label == "inst_res":
            print(
                f'{label}\t = {scaled_mean[i] * int(args.resolution):.0f} ' +
                f'+/- {scaled_std[i] * int(args.resolution):.0f}'
            )
        elif label in ["rv", "log_vmacro", "log_vsini"]:
            print(
                f'{label}\t = {scaled_mean[i]:.2f} +/- {scaled_std[i]:.2f}'
            )
        else:
            print(
                f'[{label}/Fe]\t = {unscaled_mean[i] - unscaled_mean[payne.labels.index("Fe")]:.4f} ' +
                f'+/- {np.sqrt(unscaled_std[i] ** 2 + unscaled_std[payne.labels.index("Fe")] ** 2):.4f} ' +
                f'({scaled_mean[i]:.4f} +/- {unscaled_std[i]:.4f})'
            )

    ##############################
    ######## PLOT SAMPLES ########
    ##############################
    if configs['output']['plot_chains']:
        fig, axes = plt.subplots(len(label_names), figsize=(10, 15), sharex=True)
        for i in range(len(label_names)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(label_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(fig_dir.joinpath(f"{obs_name}_chains_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.png"))
    if configs['output']['plot_corner']:
        fig = corner(flat_samples, labels=label_names)
        fig.savefig(fig_dir.joinpath(f"{obs_name}_corner_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.png"))
