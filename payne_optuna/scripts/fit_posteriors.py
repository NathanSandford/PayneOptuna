import argparse
import yaml
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from itertools import chain

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import truncnorm

from payne_optuna.fitting import PayneStitchedEmulator
from payne_optuna.misc import model_io
from payne_optuna.misc.sampling import MCMCReader, randdist

import matplotlib as mpl
import matplotlib.pyplot as plt


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Fit Observed Spectrum w/ Optimizer")
    parser.add_argument("program", help="Program")
    parser.add_argument('-o', '--overwrite', help="overwrite previous posterior fits")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


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
    # I/O Prep
    payne_wrkdir = Path('/global/scratch/users/nathan_sandford/payne_wrkdir')
    data_dir = payne_wrkdir.joinpath('data')
    mcmc_dir = data_dir.joinpath(f'{args.program}/samples')
    fig_dir = data_dir.joinpath('posterior_figures')
    program_config_dir = payne_wrkdir.joinpath(f"jobs/fitting/{args.program}/configs")
    program_configs = sorted(
        list(program_config_dir.glob('*_*_*_???_?????_bin*.yml')) +
        list(program_config_dir.glob('*_*_*_???_default.yml'))
    )
    ###############################################
    ######## Read Label Bounds from Priors ########
    ###############################################
    example_configs = yaml.load(open(program_configs[-1]), Loader=yaml.FullLoader)
    payne = model_io.load_minimal_emulator(example_configs, PayneStitchedEmulator)
    priors = model_io.get_priors(payne, example_configs)
    label_names_ = deepcopy(payne.labels)
    if example_configs['fitting']['fit_inst_res']:
        label_names_.append("inst_res")
    if example_configs['fitting']['fit_vsini']:
        label_names_.append("log_vsini")
    if example_configs['fitting']['fit_vmacro']:
        label_names_.append("log_vmacro")
    label_names_.append("rv")
    label_names = deepcopy(label_names_)
    label_names[label_names.index('log_vmacro')] = 'vmacro'
    try:
        pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'bounds')
        bounds_data_dne = False
    except (FileNotFoundError, KeyError):
        bounds_data_dne = True
    if args.overwrite or bounds_data_dne:
        lower_bounds = {payne.labels[i]: priors['stellar_labels'][i].lower_bound.item() for i in
                        range(payne.n_stellar_labels)}
        upper_bounds = {payne.labels[i]: priors['stellar_labels'][i].upper_bound.item() for i in
                        range(payne.n_stellar_labels)}
        lower_bounds['vmacro'] = 10 ** -1.0
        upper_bounds['vmacro'] = 10 ** +1.3
        lower_bounds['rv'] = -300
        upper_bounds['rv'] = +300
        bounds = pd.DataFrame(
            [lower_bounds, upper_bounds],
            index=['lower_bounds', 'upper_bounds'],
        )
        bounds.to_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'bounds')
    else:
        bounds = pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'bounds')
        lower_bounds = bounds.loc['lower_bounds']
        upper_bounds = bounds.loc['upper_bounds']
    ###################################
    ######## Prepare Dataframe ########
    ###################################
    try:
        pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'individual')
        pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'stacked')
        posterior_data_dne = False
    except (FileNotFoundError, KeyError):
        posterior_data_dne = True
    if args.overwrite or posterior_data_dne:
        mcmc_files = sorted(list(mcmc_dir.glob(f'*.h5')))
        mcmc_files = [mcmc_file for mcmc_file in mcmc_files if mcmc_file.name != 'test.h5']
        programs = [mcmc_file.parents[1].name for mcmc_file in mcmc_files]
        stars = [mcmc_file.name.split('_')[0] for mcmc_file in mcmc_files]
        frames = [mcmc_file.name.split('_')[1] for mcmc_file in mcmc_files]
        dates = [mcmc_file.name.split('_')[2] for mcmc_file in mcmc_files]
        tmp_resolutions = [mcmc_file.name.split('_')[3] for mcmc_file in mcmc_files]
        default_flag = [True if res == 'default' else False for res in tmp_resolutions]
        resolutions = [int(res) if res != 'default' else example_configs['observation']['default_res'] for res in tmp_resolutions]
        sample_method = [mcmc_file.name.split('_')[4] if 'snr' in mcmc_file.name else mcmc_file.name.split('_')[4][:-3]
                              for mcmc_file in mcmc_files]
        snr_factor = [int(mcmc_file.name.split('_')[5][3:5]) if 'snr' in mcmc_file.name else 1 for mcmc_file in
                           mcmc_files]
        exp_tags = [f'{programs[i]}_{stars[i]}_{dates[i]}_{frames[i]}' for i in range(len(mcmc_files))]
        obs_tags = [f'{programs[i]}_{stars[i]}_{resolutions[i]:05}_{sample_method[i]}_snr{snr_factor[i]:02}' for i in range(len(mcmc_files))]
        mcmc_df = pd.DataFrame(index=[file.name for file in mcmc_files])
        mcmc_df['exp_tag'] = exp_tags
        mcmc_df['obs_tag'] = obs_tags
        mcmc_df['reference'] = 'Sandford+ 2021'
        mcmc_df['program'] = programs
        mcmc_df['star'] = stars
        mcmc_df['frame'] = frames
        mcmc_df['date'] = dates
        mcmc_df['resolution'] = resolutions
        mcmc_df['samp_method'] = sample_method
        mcmc_df['snr_factor'] = snr_factor
        mcmc_df['default_res'] = default_flag
        mcmc_df['mcmc_file'] = [str(file) for file in mcmc_files]
        stack_df = pd.DataFrame(index=mcmc_df['obs_tag'].unique())
    else:
        mcmc_df = pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'individual')
        stack_df = pd.read_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'stacked')
    ##############################
    ######## Read Samples ########
    ##############################
    try:
        np.load(data_dir.joinpath(f'{args.program}_mcmc_samples.npz'))
        sample_data_dne = False
    except FileNotFoundError:
        sample_data_dne = True
    if args.overwrite or sample_data_dne:
        unscaled_samples_dict = {}
        for obs_tag in tqdm(mcmc_df['obs_tag'].unique()):
            exp_df = mcmc_df[mcmc_df['obs_tag'] == obs_tag]
            reader_dict = {}
            for exp in tqdm(exp_df.index):
                program = exp_df.loc[exp, 'program']
                # Median S/N
                if 'snr' in exp:
                    star, frame, date, resolution, samp_method, snr_factor = exp.split('_')
                else:
                    star, frame, date, resolution, samp_method = exp.split('_')
                    samp_method = samp_method[:-3]
                optim_dir = data_dir.joinpath(f"{program}/fits/{star}_{date}")
                opt_file = optim_dir.joinpath(f'{star}_{frame}_{date}_fit_{resolution}_{samp_method}_1.npz')
                optim_fit = np.load(opt_file, allow_pickle=True)
                mcmc_df.loc[exp, 'median_snr'] = np.median(
                    optim_fit['obs_flux'][optim_fit['obs_mask']] / optim_fit['obs_errs'][optim_fit['obs_mask']]
                ) / mcmc_df.loc[exp, 'snr_factor']
                # MCMC Samples
                mcmc_file = exp_df['mcmc_file'].loc[exp]
                reader = MCMCReader(mcmc_file, payne, label_names)
                reader_dict[exp] = reader
            n_samples = np.min(
                [reader.unscaled_samples.shape[0] for exp, reader in reader_dict.items()]
            )
            unscaled_samples_dict[obs_tag] = np.array(
                [
                    reader.unscaled_samples[
                        np.random.choice(
                            reader.unscaled_samples.shape[0],
                            size=n_samples,
                            replace=False
                        )
                    ] for exp, reader in reader_dict.items()
                ]
            )
        # Save Thinned Samples
        np.savez(data_dir.joinpath(f'{args.program}_mcmc_samples.npz'), **unscaled_samples_dict)
    else:
        unscaled_samples_dict = np.load(data_dir.joinpath(f'{args.program}_mcmc_samples.npz'))
    ##############################
    ######## Fit Posteriors ######
    ##############################
    if args.overwrite:
        skip = {obs_tag: [] for obs_tag in stack_df.index}
    else:
        skip = {obs_tag: [label for label in label_names if not np.isfinite(stack_df.loc[obs_tag, label])]
                for obs_tag in stack_df.index}
    for obs_tag in tqdm(mcmc_df['obs_tag'].unique()):
        for i, label in enumerate(tqdm(label_names)):
            if label in ['Teff', 'logg']:
                continue
            print(f'{obs_tag}_{label}')
            data = unscaled_samples_dict[obs_tag][:, :, i]
            x = np.linspace(data.min() - 0.1, data.max() + 0.1, 10000)
            dx = np.diff(x)[0]
            n_exp = data.shape[0]
            ### INDIVIDUAL POSTERIOR FITS ###
            model = pm.Model()
            with model:
                # Priors
                mu_exp = pm.TruncatedNormal(
                    "mu_exp",
                    mu=np.mean(data, axis=(0, 1)),
                    sigma=upper_bounds[label] - lower_bounds[label],
                    lower=lower_bounds[label],
                    upper=upper_bounds[label],
                    shape=(n_exp,)
                )
                sigma_exp = pm.HalfNormal("sigma_exp", 0.2, shape=(n_exp,))
                # Likelihood
                for j in range(n_exp):
                    theta_exp = pm.TruncatedNormal(
                        f"theta_exp_{j}",
                        mu=mu_exp[j],
                        sigma=sigma_exp[j],
                        lower=lower_bounds[label],
                        upper=upper_bounds[label],
                        observed=data[j, :],
                    )
                # Sample
                trace = pm.sample(
                    draws=2000,
                    tune=2000,
                    initvals={
                        'mu_exp': np.mean(data, axis=1),
                        'sigma_exp': np.std(data, axis=1),
                    },
                    init='adapt_diag',
                    target_accept=0.9,
                )
            # Save
            mu_exp_med = trace.posterior['mu_exp'].median(axis=(0, 1))
            sigma_exp_med = trace.posterior['sigma_exp'].median(axis=(0, 1))
            mcmc_df.loc[mcmc_df.index[mcmc_df['obs_tag'] == obs_tag], label] = mu_exp_med
            mcmc_df.loc[mcmc_df.index[mcmc_df['obs_tag'] == obs_tag], label + '_err'] = sigma_exp_med
            # SAMPLE FROM STACKED POSTERIOR
            p = np.zeros((n_exp, len(x)))
            stacked_p = np.ones_like(x)
            for j in range(n_exp):
                a = (lower_bounds[label] - mu_exp_med[j]) / sigma_exp_med[j]
                b = (upper_bounds[label] - mu_exp_med[j]) / sigma_exp_med[j]
                dist = truncnorm(loc=mu_exp_med[j], scale=sigma_exp_med[j], a=a, b=b)
                p[j] = dist.pdf(x)
                stacked_p *= p[j]
            stacked_p /= dx / 2 * (2 * stacked_p.sum() - stacked_p[0] - stacked_p[-1])
            sample_from_stacked_p = randdist(x, stacked_p, 10000)
            sample_from_stacked_p = sample_from_stacked_p[
                (sample_from_stacked_p > lower_bounds[label]) &
                (sample_from_stacked_p < upper_bounds[label])
                ]
            ### STACKED POSTERIOR FIT ###
            model = pm.Model()
            with model:
                # Priors
                mu_stack = pm.TruncatedNormal(
                    "mu_stack",
                    mu=np.array(np.mean(mu_exp_med)),
                    sigma=np.array(np.mean(sigma_exp_med)),
                    lower=lower_bounds[label],
                    upper=upper_bounds[label],
                )
                sigma_stack = pm.HalfNormal("sigma_stack", 0.2)
                # Likelihood
                TruncatedNormal = pm.TruncatedNormal(
                    'theta_stack',
                    mu=mu_stack,
                    sigma=sigma_stack,
                    lower=lower_bounds[label],
                    upper=upper_bounds[label],
                    observed=sample_from_stacked_p,
                )
                # Sample
                trace = pm.sample(
                    draws=2000,
                    tune=2000,
                    initvals={
                        'mu_stack': np.array(np.mean(mu_exp_med)),
                        'sigma_stack': np.array(np.mean(sigma_exp_med)) / np.sqrt(n_exp),
                    },
                    init='adapt_diag',
                    target_accept=0.9,
                )
            # Save
            mu_stack_med = trace.posterior['mu_stack'].median()
            sigma_stack_med = trace.posterior['sigma_stack'].median()
            stack_df.loc[obs_tag, label] = mu_stack_med
            stack_df.loc[obs_tag, label + '_err'] = sigma_stack_med
            ### PLOT FITS ###
            c = plt.cm.get_cmap('Dark2', n_exp)
            bins = np.histogram_bin_edges(data.flatten(), 'auto')
            hist = np.zeros((data.shape[0], bins.shape[0] - 1))
            stack_hist = np.ones_like(x)
            plt.figure(figsize=(10, 5))
            for j in range(n_exp):
                hist[j], _ = np.histogram(data[j], bins=bins, density=True)
                plt.plot(
                    x,
                    p[j],
                    c=c(j),
                    ls='-',
                    alpha=0.5,
                )
                plt.hist(data[j, :], bins=bins, histtype='step', color=c(j), ls='--', alpha=0.9, density=True)
            hist_prod = np.prod(hist, axis=0) / (np.prod(hist, axis=0) * np.diff(bins)).sum()
            a = (lower_bounds[label] - mu_stack_med) / sigma_stack_med
            b = (upper_bounds[label] - mu_stack_med) / sigma_stack_med
            dist = truncnorm(loc=mu_stack_med, scale=sigma_stack_med, a=a, b=b)
            stacked_p_fit = dist.pdf(x)
            plt.stairs(
                hist_prod,
                bins,
                color='k',
                ls='--',
                lw=2,
                alpha=0.9
            )
            plt.plot(
                x,
                stacked_p_fit,
                c='k',
                ls='-',
                alpha=0.75,
            )
            plt.plot(
                x,
                stacked_p,
                c='r',
                ls=':',
            )
            plt.xlim(data.min() - 0.01, data.max() + 0.01)
            plt.savefig(fig_dir.joinpath(f'{obs_tag}_{label}.png'))
            # Save Fits (checkpoint)
            mcmc_df.to_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'individual')
            stack_df.to_hdf(data_dir.joinpath(f'{args.program}_mcmc_summary.h5'), 'stacked')