import argparse
import yaml
from pathlib import Path
from copy import deepcopy

import numpy as np

import torch
from payne_optuna.fitting import PayneOrderEmulator
from payne_optuna.fitting import UniformLogPrior, GaussianLogPrior, FlatLogPrior, gaussian_log_likelihood
from payne_optuna.utils import ensure_tensor
from payne_optuna.misc import model_io

import emcee

import matplotlib as mpl
import matplotlib.pyplot as plt
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


def clip_p0(p0_, label_names, prior_list, model):
    fe_idx = label_names.index('Fe')
    fe_scaled_idx = [
        i for i, label in enumerate(label_names)
        if label not in ['Teff', 'logg', 'v_micro', 'Fe', 'inst_res', 'log_vsini', 'log_vmacro', 'rv']
    ]
    fe_scaler = torch.zeros(2, len(label_names))
    fe_scaler[:, fe_scaled_idx] = 1
    stellar_label_bounds = torch.Tensor([
                [prior.lower_bound, prior.upper_bound]
                if type(prior) == UniformLogPrior
                else [-np.inf, np.inf]
                for prior in prior_list
            ]).T
    p0_clipped = np.zeros_like(p0_)
    for i in range(p0_.shape[0]):
        unscaled_stellar_labels = model.unscale_stellar_labels(ensure_tensor(p0_[i, :model.n_stellar_labels]))
        label_bounds = stellar_label_bounds + fe_scaler * unscaled_stellar_labels[fe_idx]
        scaled_stellar_label_bounds = model.scale_stellar_labels(
            label_bounds[:, :model.n_stellar_labels]
        )
        scaled_label_bounds = np.hstack([scaled_stellar_label_bounds, label_bounds[:, model.n_stellar_labels:]])
        p0_clipped[i] = p0_[i].clip(
            min=scaled_label_bounds[0]+1e-4,
            max=scaled_label_bounds[1]-1e-4,
        )
    return p0_clipped


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
    bin_errors = configs['observation']['bin_errors']
    snr_rdx = configs['observation']['snr_rdx']
    snr_tag = f'_snr{snr_rdx:02.0f}' if snr_rdx is not False else ''
    obs_name = f'{star}_{frame}_{date}'
    # I/O Prep
    model_config_dir = Path(configs['paths']['model_config_dir'])
    data_dir = Path(configs['paths']['data_dir'])
    program_dir = data_dir.joinpath(f'{program}')
    flats_dir = program_dir.joinpath(f'flats/{date}')
    obs_dir = program_dir.joinpath(f'obs/{star}_{date}')
    mask_dir = program_dir.joinpath('masks')
    fig_dir = program_dir.joinpath(f'figures/{star}_{date}')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = program_dir.joinpath(f'fits/{star}_{date}')
    fits_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = program_dir.joinpath('samples')
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

    ######################
    ######## DATA ########
    ######################
    ## Load Flats
    #print(f'Loading flats')
    #flats = {
    #    float(flat_file.name[-6]): hires.MasterFlats(flat_file)
    #    for flat_file in flat_files
    #}
    ## Load Spectra
    #print(f'Loading {obs_spec_file.name}')
    #if isinstance(orders, str) and orders.lower() == 'all':
    #    orders_to_fit = hires.get_all_order_numbers(obs_spec_file)
    #elif np.any(np.array(orders, dtype=int) < 0):
    #    orders_to_fit = hires.get_all_order_numbers(obs_spec_file)
    #    for order in orders:
    #        orders_to_fit = np.delete(orders_to_fit, np.argwhere(np.abs(order) == orders_to_fit))
    #else:
    #    orders_to_fit = orders
    #obs = hires.load_spectrum(
    #    spec_file=obs_spec_file,
    #    orders_to_fit=orders_to_fit,
    #    extraction='OPT',
    #    flats=flats,
    #    vel_correction='heliocentric',
    #)
    ## Save Raw Errors
    #obs['raw_errs'] = deepcopy(obs['errs'])

    #######################
    ######## MASKS ########
    #######################
    #print('Generating masks')
    ## Overlapping Orders
    #obs['ovrlp_mask'] = hires.get_ovrlp_mask(obs=obs)
    #if detector_mask_file is not False:
    #    with open(detector_mask_file) as file:
    #        detector_masks = yaml.load(file, Loader=yaml.FullLoader)
    #    # Detector Edges & Bad Detector Response
    #    obs['det_mask'] = hires.get_det_mask(
    #        obs=obs,
    #        mask_left_pixels=detector_masks['detector_edge_left'],
    #        mask_right_pixels=detector_masks['detector_edge_right'],
    #        masks_by_order_wave=detector_masks['by_order_wave']
    #    )
    #    # Bad Pixels
    #    if detector_masks['bad_pixel_search'] is not False:
    #        obs['pix_mask'] = hires.get_pix_mask(
    #            obs=obs,
    #            pos_excess=detector_masks['bad_pixel_search']['pos_excess'],
    #            neg_excess=detector_masks['bad_pixel_search']['neg_excess'],
    #            window=detector_masks['bad_pixel_search']['window'],
    #            percentile=detector_masks['bad_pixel_search']['percentile'],
    #            exclude_orders=detector_masks['bad_pixel_search']['excluded_orders']
    #        )
    #    else:
    #        obs['pix_mask'] = deepcopy(obs['mask'])
    #else:
    #    obs['det_mask'] = np.ones_like(obs['mask'], dtype=bool)
    #    obs['pix_mask'] = deepcopy(obs['mask'])
    ## Telluric Lines
    #obs['tell_mask'] = hires.get_tell_mask(obs=obs, telluric_file=telluric_file)
    ## Combine Masks
    #pre_conv_mask = (obs['det_mask'] & obs['pix_mask'] & obs['tell_mask'])
    #post_conv_mask = obs['ovrlp_mask']
    #all_masks = (pre_conv_mask & post_conv_mask)
    #obs['tot_mask'] = all_masks
    #obs['errs'][~all_masks] = np.inf

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
        if isinstance(orders, str) and orders.lower() == 'all':
            pass
        elif np.any(np.array(orders, dtype=int) < 0):
            if str(int(np.abs(orders))) in str(model_config_file):
                print(f'Skipping Model {model_config_file.name[:-4]}')
                continue
        model = model_io.load_model(model_config_file)
        #  Mask Lines
        model_io.mask_lines(model, line_masks, mask_value=1.0)
        # Mask NLTE Lines
        model.mod_errs = np.sqrt(model.mod_errs ** 2 + nlte_errs[i] ** 2)
        models.append(model)
    # Sort Models by Ascending Wavelength
    models = [models[i] for i in np.argsort([model.wavelength.min() for model in models])]
    # Determine Model Breaks
    #model_bounds = model_io.find_model_breaks(models, obs)
    #print('Model bounds determined to be:')
    #[print(f'{i[0]:.2f} - {i[1]:.2f} Angstrom') for i in model_bounds]
    # Initialize Emulator
    payne = PayneOrderEmulator(
        models=models,
        cont_deg=6,
        cont_wave_norm_range=(-10, 10),
        obs_wave=obs['wave'],
        obs_blaz=obs['blaz'],
        include_model_errs=True,
        model_res=default_res,
        vmacro_method='iso_fft',
    )

    #############################
    ######## CONVOLUTION ########
    #############################
    ## Convolve Observed Spectrum
    #if resolution != "default":
    #    print(f'Convolving Observed Spectrum to R={resolution}')
    #    # Interpolate over Bad Pixels
    #    masked_spec = deepcopy(obs['spec'])
    #    for i, order in enumerate(obs['ords']):
    #        masked_spec[i][~obs['pix_mask'][i]] = np.interp(
    #            obs['wave'][i][~obs['pix_mask'][i]],
    #            obs['wave'][i][obs['pix_mask'][i]],
    #            obs['spec'][i][obs['pix_mask'][i]],
    #        )
    #    # Convolve Flux and Errors
    #    conv_obs_flux, conv_obs_errs = payne.inst_broaden(
    #        wave=ensure_tensor(obs['wave'], precision=torch.float64),
    #        flux=ensure_tensor(masked_spec).unsqueeze(0),
    #        errs=ensure_tensor(obs['raw_errs']).unsqueeze(0),
    #        inst_res=ensure_tensor(int(resolution)),
    #        model_res=ensure_tensor(default_res),
    #    )
    #
    #    # Convolve Blaze Function
    #    conv_obs_blaz, _ = payne.inst_broaden(
    #        wave=ensure_tensor(obs['wave'], precision=torch.float64),
    #        flux=ensure_tensor(obs['scaled_blaz']).unsqueeze(0),
    #        errs=None,
    #        inst_res=ensure_tensor(int(resolution)),
    #        model_res=ensure_tensor(default_res),
    #    )
    #    # Convolve Mask
    #    conv_obs_mask, _ = payne.inst_broaden(
    #        wave=ensure_tensor(obs['wave'], precision=torch.float64),
    #        flux=ensure_tensor(pre_conv_mask).unsqueeze(0),
    #        errs=None,
    #        inst_res=ensure_tensor(int(resolution)),
    #        model_res=ensure_tensor(default_res),
    #    )
    #    # Downsample
    #    ds = int(default_res/resolution)
    #    n_ord = obs['wave'].shape[0]
    #    n_pix = int(obs['wave'].shape[1]/ds)
    #    wave_ds = np.zeros((n_ord, n_pix))
    #    spec_ds = np.zeros((n_ord, n_pix))
    #    errs_ds = np.zeros((n_ord, n_pix))
    #    blaz_ds = np.zeros((n_ord, n_pix))
    #    mask1_ds = np.zeros((n_ord, n_pix))
    #    mask2_ds = np.zeros((n_ord, n_pix))
    #    for i in range(n_ord):
    #        wave_ds[i] = np.mean([obs['wave'][i, 0::ds], obs['wave'][i, ds-1::ds]], axis=0)
    #        spec_ds[i], errs_ds[i] = spectres(
    #            wave_ds[i],
    #            obs['wave'][i],
    #            conv_obs_flux[0, i].detach().numpy(),
    #            conv_obs_errs[0, i].detach().numpy(),
    #            fill_flux=conv_obs_flux[0, i,  -1].detach().numpy(),
    #            fill_errs=conv_obs_errs[0, i, -1].detach().numpy(),
    #            bin_errs=bin_errors,
    #            verbose=False,
    #        )
    #        blaz_ds[i] = spectres(
    #            wave_ds[i],
    #            obs['wave'][i],
    #            conv_obs_blaz[0, i].detach().numpy(),
    #            fill_flux=conv_obs_blaz[0, i, -1].detach().numpy(),
    #            bin_errs=bin_errors,
    #            verbose=False,
    #        )
    #        mask1_ds[i], mask2_ds[i] = spectres(
    #            wave_ds[i],
    #            obs['wave'][i],
    #            conv_obs_mask[0, i].detach().numpy(),
    #            post_conv_mask[i].astype(float),
    #            fill_flux=conv_obs_mask[0, i, -1].detach().numpy(),
    #            fill_errs=post_conv_mask[i, -1].astype(float),
    #            bin_errs=False,
    #            verbose=False
    #        )
    #    mask_ds = (mask1_ds > 0.99) & (mask2_ds > 0.99)
    #    errs_ds[~mask_ds] = np.inf
    #    obs['conv_wave'] = wave_ds
    #    obs['conv_spec'] = spec_ds
    #    obs['conv_errs'] = errs_ds
    #    obs['conv_blaz'] = blaz_ds
    #    obs['conv_mask'] = mask_ds
    #    payne.set_obs_wave(wave_ds)
    #    payne.obs_blaz = ensure_tensor(blaz_ds)
    #else:
    #    print('Using default resolution')

    ###############################
    ######## PLOT SPECTRUM ########
    ###############################
    # Plot Observed Spectrum & Blaze Function
    #if configs['output']['plot_obs']:
    #    print('Plotting 1D spectrum and blaze function')
    #    n_ord = obs['ords'].shape[0]
    #    fig = plt.figure(figsize=(10, n_ord))
    #    gs = GridSpec(n_ord, 1)
    #    gs.update(hspace=0.5)
    #    for j, order in enumerate(obs['ords']):
    #        ax = plt.subplot(gs[j, 0])
    #        ax.plot(obs['wave'][j], obs['scaled_blaz'][j], alpha=0.8, c='r', label='Scaled Blaze')
    #        ax.scatter(
    #            obs['wave'][j],
    #            obs['spec'][j],
    #            alpha=0.8, marker='.', s=1, c='k', label='Observed Spectrum'
    #        )
    #        if resolution != "default":
    #            ax.plot(obs['conv_wave'][j], obs['conv_blaz'][j], alpha=0.8, c='r', ls='--')
    #            ax.scatter(
    #                obs['conv_wave'][j],
    #                obs['conv_spec'][j],
    #                alpha=0.8, marker='.', s=1, c='b', label='Convolved Spectrum'
    #            )
    #            for k, conv_mask_range in enumerate(find_runs(0.0, obs['conv_mask'][j])):
    #                label = 'Convolved Mask' if k == 0 else ''
    #                ax.axvspan(
    #                    obs['conv_wave'][j, conv_mask_range[0]],
    #                    obs['conv_wave'][j, np.min([len(obs['conv_wave'][j])-1, conv_mask_range[1]])],
    #                    color='grey', alpha=0.2, label=label,
    #                )
    #        for k, ovrlp_mask_range in enumerate(find_runs(0.0, obs['ovrlp_mask'][j])):
    #            label = 'Overlapping Regions' if k == 0 else ''
    #            ax.axvspan(
    #                obs['wave'][j, ovrlp_mask_range[0]],
    #                obs['wave'][j, np.min([len(obs['wave'][j])-1, ovrlp_mask_range[1]])],
    #                color='pink', alpha=0.25, label=label,
    #            )
    #        for k, det_mask_range in enumerate(find_runs(0.0, obs['det_mask'][j])):
    #            label = 'Detector Mask' if k == 0 else ''
    #            ax.axvspan(
    #                obs['wave'][j, det_mask_range[0]],
    #                obs['wave'][j, np.min([len(obs['wave'][j])-1, det_mask_range[1]])],
    #                color='grey', alpha=0.5, label=label,
    #            )
    #        for k, tell_mask_range in enumerate(find_runs(0.0, obs['tell_mask'][j])):
    #            label = 'Telluric Mask' if k == 0 else ''
    #            ax.axvspan(
    #                obs['wave'][j, tell_mask_range[0]],
    #                obs['wave'][j, np.min([len(obs['wave'][j])-1, tell_mask_range[1]])],
    #                color='skyblue', alpha=0.5, label=label,
    #            )
    #        for k, bad_pix_range in enumerate(find_runs(0.0, obs['pix_mask'][j])):
    #            label = 'Bad Pixel Mask' if k == 0 else ''
    #            ax.axvspan(
    #                obs['wave'][j, bad_pix_range[0]],
    #                obs['wave'][j, np.min([len(obs['wave'][j])-1, bad_pix_range[1]])],
    #                color='purple', alpha=0.75, label=label,
    #            )
    #        ax.set_ylim(0, 1.5 * np.quantile(obs['spec'][j][obs['mask'][j]], 0.95))
    #        ax.text(0.98, 0.70, f"Order: {int(order)}", transform=ax.transAxes, fontsize=6, verticalalignment='top',
    #                horizontalalignment='right',
    #                bbox=dict(facecolor='white', alpha=0.8))
    #        if j == 0:
    #            ax.set_title(obs_name)
    #            ax.legend(fontsize=8)
    #    plt.savefig(fig_dir.joinpath(f'{obs_name}_obs_{resolution}.png'))
    #    plt.close('all')

    ####################################
    ######## PRIORS + POSTERIOR ########
    ####################################
    # Set Priors
    print('Setting Priors')
    stellar_label_priors = []
    for i, label in enumerate(payne.labels):
        if label in configs['fitting']['priors']:
            if configs['fitting']['priors'][label][0] == 'N':
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
    #print('Setting Priors')
    #teff_mu = configs['fitting']['priors']['Teff'][0]
    #teff_sigma = configs['fitting']['priors']['Teff'][1]
    #teff_mu_scaled = payne.scale_stellar_labels(teff_mu * torch.ones(payne.n_stellar_labels))[0]
    #teff_sigma_scaled = payne.scale_stellar_labels(teff_mu * torch.ones(payne.n_stellar_labels))[0] \
    #                    - payne.scale_stellar_labels(teff_mu - teff_sigma * torch.ones(payne.n_stellar_labels))[0]
    #logg_mu = configs['fitting']['priors']['logg'][0]
    #logg_sigma = configs['fitting']['priors']['logg'][1]
    #logg_mu_scaled = payne.scale_stellar_labels(logg_mu * torch.ones(payne.n_stellar_labels))[1]
    #logg_sigma_scaled = payne.scale_stellar_labels(logg_mu * torch.ones(payne.n_stellar_labels))[1] \
    #                    - payne.scale_stellar_labels(logg_mu - logg_sigma * torch.ones(payne.n_stellar_labels))[1]
    #print(f'Teff Priors = {teff_mu:0.0f} +/- {teff_sigma:0.1f} ({teff_mu_scaled:0.2f} +/- {teff_sigma_scaled:0.2f})')
    #print(f'logg Priors = {logg_mu:0.2f} +/- {logg_sigma:0.3f} ({logg_mu_scaled:0.2f} +/- {logg_sigma_scaled:0.2f})')
    #priors = {
    #    'stellar_labels': [
    #                          GaussianLogPrior('Teff', teff_mu_scaled, teff_sigma_scaled),
    #                          GaussianLogPrior('logg', logg_mu_scaled, logg_sigma_scaled)
    #                      ] + [UniformLogPrior(lab, -0.55, 0.55) for lab in payne.labels[2:]],
    #    'log_vmacro': UniformLogPrior('log_vmacro', -1, 1.3),
    #    'log_vsini': FlatLogPrior('log_vsini'),
    #    'inst_res': FlatLogPrior('inst_res') if resolution == 'default' else
    #    GaussianLogPrior('inst_res', int(resolution), 0.01 * int(resolution))
    #}
    # Define Posterior Function
    def log_probability(theta, model, obs, priors):
        stellar_labels = theta[:, :model.n_stellar_labels]
        cont_coeffs = optim_fit["cont_coeffs"]
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
        unscaled_stellar_labels = model.unscale_stellar_labels(ensure_tensor(stellar_labels))
        fe_idx = model.labels.index('Fe')
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
    prior_list = deepcopy(priors['stellar_labels'])
    label_names = deepcopy(payne.labels)
    if configs['fitting']['fit_inst_res']:
        p0_list.append(optim_fit["inst_res"][0])
        label_names.append("inst_res")
        prior_list.append(priors['inst_res'])
    if configs['fitting']['fit_vsini']:
        p0_list.append(optim_fit["log_vsini"][0])
        label_names.append("log_vsini")
        prior_list.append(priors['log_vsini'])
    if configs['fitting']['fit_vmacro']:
        p0_list.append(optim_fit["log_vmacro"][0])
        label_names.append("log_vmacro")
        prior_list.append(priors['log_vmacro'])
    p0_list.append(optim_fit["rv"])
    label_names.append("rv")
    prior_list.append(priors['rv'])
    p0_ = np.concatenate(p0_list) + 1e-2 * np.random.randn(128, len(label_names))
    p0 = clip_p0(p0_, label_names, prior_list, payne)
    nwalkers, ndim = p0.shape
    # Initialize Backend
    sample_file = sample_dir.joinpath(f"test.h5")
    backend = emcee.backends.HDFBackend(sample_file, name=f"burn_in_1")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(payne, obs, priors),
        vectorize=True,
        backend=backend,
    )
    # Run Sampler until walkers stop wandering
    p_mean_last = p0.mean(0)
    converged = False
    for _ in sampler.sample(p0, iterations=10000, progress=True, store=True):
        if (sampler.iteration % 100):
            continue
        p_mean = sampler.get_chain(flat=True, thin=1, discard=sampler.iteration - 100).mean(0)
        print(f" ({obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag})")
        print(f'max(dMean) = {np.max(p_mean - p_mean_last):0.04f}')
        if np.abs(np.max(p_mean - p_mean_last)) < 0.001 and (sampler.iteration >= 500):
            p_mean_last = p_mean
            converged = True
            break
        p_mean_last = p_mean
    if configs['output']['plot_chains']:
        samples = sampler.get_chain()
        fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(label_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(fig_dir.joinpath(f"{obs_name}_burnin_1_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.png"))
    print('Burn-In 1 Complete')
    ### Run Burn-In 2 ###
    #if not converged:
    #    print('')
    #    # Initialize Walkers
    #    last_state = sampler.get_last_sample()
    #    best_walker = last_state.coords[last_state.log_prob.argmax()]
    #    p0 = best_walker + 1e-3 * np.random.randn(256, len(label_names))
    #    nwalkers, ndim = p0.shape
    #    # Initialize Backend
    #    backend = emcee.backends.HDFBackend(sample_file, name=f"burn_in_2")
    #    backend.reset(nwalkers, ndim)
    #    # Initialize Sampler
    #    sampler = emcee.EnsembleSampler(
    #        nwalkers,
    #        ndim,
    #        log_probability,
    #        args=(payne, obs, priors),
    #        vectorize=True,
    #        backend=backend,
    #    )
    #    # Run Sampler until walkers stop wandering
    #    for _ in sampler.sample(p0, iterations=5000, progress=True, store=True):
    #        if (sampler.iteration % 100):
    #            continue
    #        p_mean = sampler.get_chain(flat=True, thin=1, discard=sampler.iteration - 100).mean(0)
    #        print(f'max(dMean) = {np.max(p_mean - p_mean_last)}')
    #        if np.abs(np.max(p_mean - p_mean_last)) < 0.001:
    #            p_mean_last = p_mean
    #            converged = True
    #            break
    #        p_mean_last = p_mean
    #    if configs['output']['plot_chains']:
    #        samples = sampler.get_chain()
    #        fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
    #        for i in range(ndim):
    #            ax = axes[i]
    #            ax.plot(samples[:, :, i], "k", alpha=0.3)
    #            ax.set_xlim(0, len(samples))
    #            ax.set_ylabel(label_names[i])
    #            ax.yaxis.set_label_coords(-0.1, 0.5)
    #        axes[-1].set_xlabel("step number")
    #        plt.savefig(fig_dir.joinpath(f"{obs_name}_burnin_2_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}.png"))
    #    print('Burn-In 2 Complete')

    ##############################
    ######## RUN SAMPLING ########
    ##############################
    # Initialize Walkers
    last_state = sampler.get_last_sample()
    p0 = last_state.coords
    #best_walker = last_state.coords[last_state.log_prob.argmax()]
    #p0 = best_walker + 1e-3 * np.random.randn(1028, len(label_names))
    nwalkers, ndim = p0.shape
    # Initialize Backend
    backend = emcee.backends.HDFBackend(sample_file, name=f"{obs_name}")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(payne, obs, priors),
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
            discard=old_tau if np.isfinite(old_tau) else 0,
            tol=0
        )
        autocorr[index] = np.mean(tau)
        print(
            f"{obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag} " +
            f"Step {sampler.iteration}: Tau = {np.max(tau):.0f}, " +
            f"t/30Tau = {sampler.iteration / (30 * np.max(tau)):.2f}, " +
            f"|dTau/Tau| = {np.max(np.abs(old_tau - tau) / tau):.3f}"
        )
        index += 1
        # Check convergence
        converged = np.all(tau * 30 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        old_tau = tau
        if converged:
            break
        if sampler.iteration % 1000:
            continue
        # Plot Chain Update
        if configs['output']['plot_chains']:
            samples = sampler.get_chain()
            fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(label_names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            plt.savefig(fig_dir.joinpath(f"{obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_ckpt.png"))

    #################################
    ######## PROCESS SAMPLES ########
    #################################
    # Get Chains
    scaled_samples = sampler.get_chain()
    unscaled_samples = payne.unscale_stellar_labels(
        ensure_tensor(
            scaled_samples[:, :, :payne.n_stellar_labels].reshape(-1, payne.n_stellar_labels)
        )
    ).reshape(-1, nwalkers, payne.n_stellar_labels).detach().numpy()
    samples = np.concatenate([
        unscaled_samples,
        scaled_samples[:, :, payne.n_stellar_labels:]
    ], axis=-1)
    # Get Flat Samples
    scaled_flat_samples = sampler.get_chain(
        discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2), flat=True
    )
    unscaled_flat_samples = payne.unscale_stellar_labels(
        ensure_tensor(scaled_flat_samples[:, :payne.n_stellar_labels])).detach().numpy()
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
                f'{label}\t = {scaled_mean[i]*int(args.resolution):.0f} ' +
                f'+/- {scaled_std[i]*int(args.resolution):.0f}'
            )
        elif label in ["rv", "log_vmacro", "log_vsini"]:
            print(
                f'{label}\t = {scaled_mean[i]:.2f} +/- {scaled_std[i]:.2f}'
            )
        else:
            print(
                f'[{label}/Fe]\t = {unscaled_mean[i] - unscaled_mean[payne.labels.index("Fe")]:.4f} ' +
                f'+/- {np.sqrt(unscaled_std[i]**2 + unscaled_std[payne.labels.index("Fe")]**2):.4f} ' +
                f'({scaled_mean[i]:.4f} +/- {unscaled_std[i]:.4f})'
            )

    ##############################
    ######## PLOT SAMPLES ########
    ##############################
    if configs['output']['plot_chains']:
        fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
        for i in range(ndim):
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
