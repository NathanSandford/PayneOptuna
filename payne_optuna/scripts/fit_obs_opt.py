import argparse
import yaml
from pathlib import Path
from copy import deepcopy
import gc

import numpy as np
import pandas as pd

import torch
from payne_optuna.fitting import PayneStitchedEmulator, PayneOptimizer, UniformLogPrior, GaussianLogPrior, FlatLogPrior
from payne_optuna.utils import ensure_tensor, find_runs, noise_up_spec
from payne_optuna.misc import hires, model_io
from payne_optuna.misc.spectres import spectres

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
    mpl.rc('figure', dpi=100)

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
    flats_dir = program_dir.joinpath(f'flats/{date}')
    obs_dir = program_dir.joinpath(f'obs/{star}_{date}')
    mask_dir = program_dir.joinpath('masks')
    fig_dir = program_dir.joinpath(f'figures/{star}_{date}')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = program_dir.joinpath(f'fits/{star}_{date}')
    fits_dir.mkdir(parents=True, exist_ok=True)
    model_config_files = sorted(list(model_config_dir.glob('*')))
    flat_files = sorted(list(flats_dir.glob('*')))
    obs_spec_file = obs_dir.joinpath(f'spec1d_{star}_{frame}.fits')
    telluric_file = mask_dir.joinpath(f"{configs['masks']['telluric']}")
    detector_mask_file = mask_dir.joinpath(f"{configs['masks']['detector']}")
    line_mask_file = mask_dir.joinpath(f"{configs['masks']['line']}") \
        if configs['masks']['line'] not in [False, None] else False
    nlte_errs_files = sorted(list(mask_dir.joinpath(configs['masks']['nlte']).glob('*'))) \
        if configs['masks']['nlte'] not in [False, None] else False

    ######################
    ######## DATA ########
    ######################
    # Load Flats
    print(f'Loading flats')
    flats = {
        float(flat_file.name[-6]): hires.MasterFlats(flat_file)
        for flat_file in flat_files
    }
    # Load Spectra
    print(f'Loading {obs_spec_file.name}')
    if isinstance(orders, str) and orders.lower() == 'all':
        orders_to_fit = hires.get_all_order_numbers(obs_spec_file)
    elif np.any(np.array(orders, dtype=int) < 0):
        orders_to_fit = hires.get_all_order_numbers(obs_spec_file)
        for order in orders:
            orders_to_fit = np.delete(orders_to_fit, np.argwhere(np.abs(order) == orders_to_fit))
    else:
        orders_to_fit = orders
    obs = hires.load_spectrum(
        spec_file=obs_spec_file,
        orders_to_fit=orders_to_fit,
        extraction='OPT',
        flats=flats,
        vel_correction='heliocentric',
    )
    # Save Raw Errors
    obs['raw_errs'] = deepcopy(obs['errs'])
    if configs['fitting']['use_gaia_phot']:
        phot = pd.read_hdf(data_dir.joinpath('photometry.h5'), 'M15')
        obs['bp-rp'] = phot.loc[star, 'bp-rp']
        obs['g'] = phot.loc[star, 'g']
        gaia_cmd_interp = np.load(data_dir.joinpath('gaia_cmd_interp.npy'), allow_pickle=True)[()]

    #######################
    ######## MASKS ########
    #######################
    print('Generating masks')
    # Overlapping Orders
    #obs['ovrlp_mask'] = hires.get_ovrlp_mask(obs=obs)  # Don't actually need to mask these regions
    obs['ovrlp_mask'] = np.ones_like(obs['mask'], dtype=bool)
    if detector_mask_file is not False:
        with open(detector_mask_file) as file:
            detector_masks = yaml.load(file, Loader=yaml.FullLoader)
        # Detector Edges & Bad Detector Response
        obs['det_mask'] = hires.get_det_mask(
            obs=obs,
            mask_left_pixels=detector_masks['detector_edge_left'],
            mask_right_pixels=detector_masks['detector_edge_right'],
            masks_by_order_wave=detector_masks['by_order_wave']
        )
        # Bad Pixels
        if detector_masks['bad_pixel_search'] is not False:
            obs['pix_mask'] = hires.get_pix_mask(
                obs=obs,
                pos_excess=detector_masks['bad_pixel_search']['pos_excess'],
                neg_excess=detector_masks['bad_pixel_search']['neg_excess'],
                window=detector_masks['bad_pixel_search']['window'],
                percentile=detector_masks['bad_pixel_search']['percentile'],
                exclude_orders=detector_masks['bad_pixel_search']['excluded_orders']
            )
        else:
            obs['pix_mask'] = deepcopy(obs['mask'])
    else:
        obs['det_mask'] = np.ones_like(obs['mask'], dtype=bool)
        obs['pix_mask'] = deepcopy(obs['mask'])
    # Telluric Lines
    obs['tell_mask'] = hires.get_tell_mask(obs=obs, telluric_file=telluric_file)
    # Combine Masks
    pre_conv_mask = (obs['det_mask'] & obs['pix_mask'] & obs['tell_mask'])
    post_conv_mask = obs['ovrlp_mask']
    all_masks = (pre_conv_mask & post_conv_mask)
    obs['tot_mask'] = all_masks

    ########################
    ######## MODELS ########
    ########################
    # Load Line Masks
    if line_mask_file not in [False, None]:
        print(f'Loading line masks')
        with open(line_mask_file) as file:
            line_masks = yaml.load(file, Loader=yaml.FullLoader)
    else:
        line_masks = None
    # Load NLTE Uncertainties
    if nlte_errs_files not in [False, None]:
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
        # Don't need to skip w/ Stitched Emulator
        #if isinstance(orders, str) and orders.lower() == 'all':
        #    pass
        #elif np.any(np.array(orders, dtype=int) < 0):
        #    if str(int(np.abs(orders))) in str(model_config_file):
        #        print(f'Skipping Model {model_config_file.name[:-4]}')
        #        continue
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
    #payne = PayneOrderEmulator(
    #    models=models,
    #    cont_deg=configs['fitting']['cont_deg'],
    #    cont_wave_norm_range=(-10, 10),
    #    obs_wave=obs['wave'],
    #    obs_blaz=obs['scaled_blaz'],
    #    include_model_errs=True,
    #    model_res=default_res,
    #    vmacro_method='iso_fft',
    #)
    payne = PayneStitchedEmulator(
        models=models,
        cont_deg=configs['fitting']['cont_deg'],
        cont_wave_norm_range=(-10, 10),
        obs_wave=obs['wave'],
        obs_blaz=obs['scaled_blaz'],
        include_model_errs=True,
        model_res=model_res,
        vmacro_method='iso_fft',
    )

    #############################
    ######## CONVOLUTION ########
    #############################
    # Convolve Observed Spectrum
    if resolution != "default":
        print(f'Convolving Observed Spectrum to R={resolution}')
        # Interpolate over Bad Pixels
        masked_spec = deepcopy(obs['spec'])
        for i, order in enumerate(obs['ords']):
            masked_spec[i][~obs['pix_mask'][i]] = np.interp(
                obs['wave'][i][~obs['pix_mask'][i]],
                obs['wave'][i][obs['pix_mask'][i]],
                obs['spec'][i][obs['pix_mask'][i]],
            )
        # Mask Potential Off-detector Lines
        pre_conv_mask[:, [0, -1]] = False
        # Convolve Flux and Errors
        conv_obs_flux, conv_obs_errs = payne.inst_broaden(
            wave=ensure_tensor(obs['wave'], precision=torch.float64),
            flux=ensure_tensor(masked_spec).unsqueeze(0),
            errs=ensure_tensor(obs['raw_errs']).unsqueeze(0),
            inst_res=ensure_tensor(int(resolution)),
            model_res=ensure_tensor(default_res),
        )

        # Convolve Blaze Function
        conv_obs_blaz, _ = payne.inst_broaden(
            wave=ensure_tensor(obs['wave'], precision=torch.float64),
            flux=ensure_tensor(obs['scaled_blaz']).unsqueeze(0),
            errs=None,
            inst_res=ensure_tensor(int(resolution)),
            model_res=ensure_tensor(default_res),
        )
        # Convolve Mask
        conv_obs_mask, _ = payne.inst_broaden(
            wave=ensure_tensor(obs['wave'], precision=torch.float64),
            flux=ensure_tensor(pre_conv_mask).unsqueeze(0),
            errs=None,
            inst_res=ensure_tensor(int(resolution)),
            model_res=ensure_tensor(default_res),
        )
        # Downsample
        ds = int(np.round(default_res/resolution))
        n_ord = obs['wave'].shape[0]
        n_pix = int(obs['wave'].shape[1]/ds)
        wave_ds = np.zeros((n_ord, n_pix))
        spec_ds = np.zeros((n_ord, n_pix))
        errs_ds = np.zeros((n_ord, n_pix))
        blaz_ds = np.zeros((n_ord, n_pix))
        mask1_ds = np.zeros((n_ord, n_pix))
        mask2_ds = np.zeros((n_ord, n_pix))
        for i in range(n_ord):
            wave_ds[i] = np.mean([obs['wave'][i, 0::ds], obs['wave'][i, ds-1::ds]], axis=0)
            spec_ds[i], errs_ds[i] = spectres(
                wave_ds[i],
                obs['wave'][i],
                conv_obs_flux[0, i].detach().numpy(),
                conv_obs_errs[0, i].detach().numpy(),
                fill_flux=conv_obs_flux[0, i,  -1].detach().numpy(),
                fill_errs=conv_obs_errs[0, i, -1].detach().numpy(),
                bin_errs=bin_errors,
                verbose=False,
            )
            blaz_ds[i] = spectres(
                wave_ds[i],
                obs['wave'][i],
                conv_obs_blaz[0, i].detach().numpy(),
                fill_flux=conv_obs_blaz[0, i, -1].detach().numpy(),
                bin_errs=bin_errors,
                verbose=False,
            )
            mask1_ds[i], mask2_ds[i] = spectres(
                wave_ds[i],
                obs['wave'][i],
                conv_obs_mask[0, i].detach().numpy(),
                post_conv_mask[i].astype(float),
                fill_flux=conv_obs_mask[0, i, -1].detach().numpy(),
                fill_errs=post_conv_mask[i, -1].astype(float),
                bin_errs=False,
                verbose=False
            )
        mask_ds = (mask1_ds > 0.99) & (mask2_ds > 0.99)
        errs_ds[~mask_ds] = 1e10
        spec_ds[~mask_ds] = 0
        obs['conv_wave'] = wave_ds
        obs['conv_spec'] = spec_ds
        obs['conv_errs'] = errs_ds
        obs['conv_blaz'] = blaz_ds
        obs['conv_mask'] = mask_ds
        payne.set_obs_wave(wave_ds)
        payne.obs_blaz = ensure_tensor(blaz_ds)
    else:
        obs['errs'][~all_masks] = 1e10
        obs['spec'][~all_masks] = 0
        print('Using default resolution')

    ######################################
    ######## DEGRADE SIGNAL/NOISE ########
    ######################################
    if snr_rdx is not False:
        print(f'Increase Noise by a Factor of {snr_rdx}')
        if resolution != "default":
            D, sigma_D = noise_up_spec(obs['conv_spec'], obs['conv_errs'], snr_rdx, seed=8645)
            obs['conv_spec'] = D
            obs['conv_errs'] = sigma_D
        else:
            D, sigma_D = noise_up_spec(obs['spec'], obs['errs'], snr_rdx, seed=8645)
            obs['spec'] = D
            obs['errs'] = sigma_D

    ###############################
    ######## PLOT SPECTRUM ########
    ###############################
    # Plot Observed Spectrum & Blaze Function
    if configs['output']['plot_obs']:
        print('Plotting 1D spectrum and blaze function')
        n_ord = obs['ords'].shape[0]
        fig = plt.figure(figsize=(10, n_ord))
        gs = GridSpec(n_ord, 1)
        gs.update(hspace=0.5)
        for j, order in enumerate(obs['ords']):
            ax = plt.subplot(gs[j, 0])
            ax.plot(obs['wave'][j], obs['scaled_blaz'][j], alpha=0.8, c='r', label='Scaled Blaze')
            ax.scatter(
                obs['wave'][j],
                obs['spec'][j],
                alpha=0.8, marker='.', s=1, c='k', label='Observed Spectrum'
            )
            if resolution != "default":
                ax.plot(obs['conv_wave'][j], obs['conv_blaz'][j], alpha=0.8, c='r', ls='--')
                ax.scatter(
                    obs['conv_wave'][j],
                    obs['conv_spec'][j],
                    alpha=0.8, marker='.', s=1, c='b', label='Convolved Spectrum'
                )
                for k, conv_mask_range in enumerate(find_runs(0.0, obs['conv_mask'][j])):
                    label = 'Convolved Mask' if k == 0 else ''
                    ax.axvspan(
                        obs['conv_wave'][j, conv_mask_range[0]],
                        obs['conv_wave'][j, np.min([len(obs['conv_wave'][j])-1, conv_mask_range[1]])],
                        color='grey', alpha=0.2, label=label,
                    )
            for k, ovrlp_mask_range in enumerate(find_runs(0.0, obs['ovrlp_mask'][j])):
                label = 'Overlapping Regions' if k == 0 else ''
                ax.axvspan(
                    obs['wave'][j, ovrlp_mask_range[0]],
                    obs['wave'][j, np.min([len(obs['wave'][j])-1, ovrlp_mask_range[1]])],
                    color='pink', alpha=0.25, label=label,
                )
            for k, det_mask_range in enumerate(find_runs(0.0, obs['det_mask'][j])):
                label = 'Detector Mask' if k == 0 else ''
                ax.axvspan(
                    obs['wave'][j, det_mask_range[0]],
                    obs['wave'][j, np.min([len(obs['wave'][j])-1, det_mask_range[1]])],
                    color='grey', alpha=0.5, label=label,
                )
            for k, tell_mask_range in enumerate(find_runs(0.0, obs['tell_mask'][j])):
                label = 'Telluric Mask' if k == 0 else ''
                ax.axvspan(
                    obs['wave'][j, tell_mask_range[0]],
                    obs['wave'][j, np.min([len(obs['wave'][j])-1, tell_mask_range[1]])],
                    color='skyblue', alpha=0.5, label=label,
                )
            for k, bad_pix_range in enumerate(find_runs(0.0, obs['pix_mask'][j])):
                label = 'Bad Pixel Mask' if k == 0 else ''
                ax.axvspan(
                    obs['wave'][j, bad_pix_range[0]],
                    obs['wave'][j, np.min([len(obs['wave'][j])-1, bad_pix_range[1]])],
                    color='purple', alpha=0.75, label=label,
                )
            ax.set_ylim(0, 1.5 * np.quantile(obs['spec'][j][obs['mask'][j]], 0.95))
            ax.text(0.98, 0.70, f"Order: {int(order)}", transform=ax.transAxes, fontsize=6, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
            if j == 0:
                ax.set_title(obs_name)
                ax.legend(fontsize=8)
        plt.savefig(fig_dir.joinpath(f'{obs_name}_obs_{resolution}{snr_tag}.png'))
        plt.close('all')

    ########################
    ######## PRIORS ########
    ########################
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
                        configs['fitting']['priors'][label][2]
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
                )
            )
    priors = {
        "stellar_labels": stellar_label_priors,
        'log_vmacro': UniformLogPrior('log_vmacro', -1, 1.3),
        'log_vsini': FlatLogPrior('log_vsini'),
        'inst_res': FlatLogPrior('inst_res') if resolution == 'default' else
        GaussianLogPrior('inst_res', int(resolution), 0.01 * int(resolution))
    }

    #############################
    ######## FIT SPECTRA ########
    #############################
    n_fits = int(configs['fitting']['n_fits'])
    for n in range(n_fits):
        print(f'Beginning fit {n + 1}/{n_fits} for {obs_name}')
        # Initialize Optimizer
        optimizer = PayneOptimizer(
            emulator=payne,
            loss_fn='neg_log_posterior',
            learning_rates=dict(
                stellar_labels=1e-1,
                log_vmacro=1e-1,
                log_vsini=1e-3,
                inst_res=1e3,
                rv=1e-3,
                cont_coeffs=1e-1,
            ),
            learning_rate_decay=dict(
                stellar_labels=0.99,
                log_vmacro=0.99,
                log_vsini=0.99,
                inst_res=0.9,
                rv=0.9,
                cont_coeffs=0.99,
            ),
            learning_rate_decay_ts=dict(
                stellar_labels=10,
                log_vmacro=10,
                log_vsini=10,
                inst_res=10,
                rv=10,
                cont_coeffs=10,
            ),
            tolerances=dict(
                d_stellar_labels=1e-5,
                d_log_vmacro=1e-4,
                d_log_vsini=np.inf,
                d_inst_res=1e0,
                d_rv=1e-4,
                d_cont=5e-2,
                d_loss=-np.inf,
                loss=-np.inf,
            ),
        )
        # Initializing Parameters
        print('Initializing Labels')
        stellar_labels0 = torch.zeros(1, payne.n_stellar_labels)
        if configs['fitting']['priors']['Fe'][0] == 'N':
            fe0 = np.random.normal(
                configs['fitting']['priors']['Fe'][1],
                configs['fitting']['priors']['Fe'][2],
            )
        elif configs['fitting']['priors']['Fe'][0] == 'U':
            fe0 = np.random.uniform(
                configs['fitting']['priors']['Fe'][1],
                configs['fitting']['priors']['Fe'][2],
            )
        else:
            raise RuntimeError("No prior provided for [Fe/H]")
        if configs['fitting']['use_gaia_phot']:
            fe_phot = np.vstack([
                fe0,
                obs['bp-rp'],
                obs['g'],
            ]).T
            logg, logTeff = gaia_cmd_interp(fe_phot).flatten()
        for i, label in enumerate(payne.labels):
            if (label == 'Teff') and configs['fitting']['use_gaia_phot']:
                x0 = payne.scale_stellar_labels(
                    10**logTeff * torch.ones(payne.n_stellar_labels)
                )[i]
            elif (label == 'logg') and configs['fitting']['use_gaia_phot']:
                x0 = payne.scale_stellar_labels(
                    logg * torch.ones(payne.n_stellar_labels)
                )[i]
            elif label in configs['fitting']['priors']:
                if configs['fitting']['priors'][label][0] == 'N':
                    x0 = np.random.normal(
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2],
                    )
                elif configs['fitting']['priors'][label][0] == 'U':
                    x0 = np.random.uniform(
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2],
                    )
                else:
                    raise KeyError(f"Cannot parse prior info for {label}")
                if label == 'Fe':
                    x0 = fe0
                elif label not in ['Teff', 'logg', 'v_micro', 'Fe']:
                    x0 += fe0
                x0 = payne.scale_stellar_labels(
                    x0 * torch.ones(payne.n_stellar_labels)
                )[i]
            else:
                x0 = np.random.uniform(
                    -0.55,
                    0.55,
                )
            with torch.no_grad():
                stellar_labels0[:, i] = ensure_tensor(x0)
        if not configs['fitting']['fit_vmicro']:
            with torch.no_grad():
                stellar_labels0[:, 2] = payne.scale_stellar_labels(
                    (2.478 - 0.325 * payne.unscale_stellar_labels(stellar_labels0)[:, 1]) * torch.ones(payne.n_stellar_labels)
                )[2]
        rv0 = 'prefit'
        log_vmacro0 = torch.FloatTensor(1, 1).uniform_(-1.0, 1.3) if configs['fitting']['fit_vmacro'] else None
        log_vsini0 = torch.FloatTensor(1, 1).uniform_(-1.0, 1.0) if configs['fitting']['fit_vsini'] else None
        if configs['fitting']['fit_inst_res']:
            inst_res0 = ensure_tensor(default_res) if resolution == 'default' else ensure_tensor(int(resolution))
            inst_res0 = torch.min(
                inst_res0 * torch.FloatTensor(1, 1).uniform_(0.9, 1.1),
                ensure_tensor(payne.model_res)
            )
            print(f'inst_res is initialized to {int(inst_res0)}')
        elif resolution != 'default':
            print(f'Fixing inst_res to {int(resolution)} (modified from observation default)')
            inst_res0 = int(resolution) * torch.ones(1, 1)
        elif (resolution == 'default') and (default_res != model_res):
            print(f'Fixing inst_res to {int(default_res)}')
            inst_res0 = int(default_res) * torch.ones(1, 1)
        elif (resolution == 'default') and (default_res == model_res):
            print('Observation has default resolution that matches model resolution---no instrumental broadening necessary')
            inst_res0 = None
        else:
            raise RuntimeError("Resolution was parsed incorrectly---this should not happen.")
        # Begin Fit
        optimizer.fit(
            obs_flux=obs['spec'] if resolution == "default" else obs['conv_spec'],
            obs_errs=obs['errs'] if resolution == "default" else obs['conv_errs'],
            obs_wave=obs['wave'] if resolution == "default" else obs['conv_wave'],
            obs_blaz=None,
            params=dict(
                stellar_labels='fit',
                rv='fit',
                log_vmacro='fit' if configs['fitting']['fit_vmacro'] else 'const',
                log_vsini='fit' if configs['fitting']['fit_vsini'] else 'const',
                inst_res='fit' if configs['fitting']['fit_inst_res'] else 'const',
                cont_coeffs='fit',
            ),
            init_params=dict(
                stellar_labels=stellar_labels0,
                rv=rv0,
                log_vmacro=log_vmacro0,
                log_vsini=log_vsini0,
                inst_res=inst_res0,
                cont_coeffs='prefit'
            ),
            priors=priors,
            use_gaia_phot=configs['fitting']['use_gaia_phot'],
            gaia_g=obs['g'] if configs['fitting']['use_gaia_phot'] else None,
            gaia_bprp=obs['bp-rp'] if configs['fitting']['use_gaia_phot'] else None,
            gaia_cmd_interpolator=gaia_cmd_interp if configs['fitting']['use_gaia_phot'] else None,
            use_holtzman2015=False if configs['fitting']['fit_vmicro'] else True,
            min_epochs=2000,
            max_epochs=50000,
            prefit_cont_window=55,
            verbose=True,
            plot_prefits=False,
            plot_fit_every=None,
        )
        # Unscale Stellar Labels
        unscaled_stellar_labels = payne.unscale_stellar_labels(optimizer.stellar_labels)
        # Print Best Fit Values
        print(f'Best Fit Labels ({n + 1}/{n_fits}):')
        for i, label in enumerate(payne.labels):
            if label not in ['Teff', 'logg', 'v_micro', 'Fe']:
                print(
                    f'[{label}/Fe]\t = {unscaled_stellar_labels[0, i] - unscaled_stellar_labels[0, payne.labels.index("Fe")]:.2f} ({optimizer.stellar_labels[0, i]:.2f})')
            elif label == 'Fe':
                print(f'[{label}/H]\t = {unscaled_stellar_labels[0, i]:.2f} ({optimizer.stellar_labels[0, i]:.2f})')
            else:
                print(f'{label}\t = {unscaled_stellar_labels[0, i]:.2f} ({optimizer.stellar_labels[0, i]:.2f})')
        print(f'RV\t = {optimizer.rv.item():.2f} ({payne.rv_scale} km/s)')
        if optimizer.log_vmacro is not None:
            print(f'vmacro\t = {10**optimizer.log_vmacro.item():.2f} (km/s)')
        if optimizer.log_vsini is not None:
            print(f'vsini\t = {10**optimizer.log_vsini.item():.2f} (km/s)')
        if optimizer.inst_res is not None:
            print(f'inst_res\t = {optimizer.inst_res.item():.2f} (km/s)')
        # Save Labels & Fits
        optim_fit = {
            'stellar_labels': optimizer.stellar_labels.detach(),
            'rv': optimizer.rv.detach(),
            'log_vmacro': optimizer.log_vmacro.detach() if optimizer.log_vmacro is not None else None,
            'log_vsini': optimizer.log_vsini.detach() if optimizer.log_vsini is not None else None,
            'inst_res': optimizer.inst_res.detach() if optimizer.inst_res is not None else None,
            'cont_coeffs': torch.stack(optimizer.cont_coeffs).detach(),
            'obs_wave': optimizer.obs_wave.detach(),
            'obs_flux': optimizer.obs_flux.detach(),
            'obs_errs': optimizer.obs_errs.detach(),
            'obs_blaz': optimizer.emulator.obs_blaz.detach(),
            'obs_mask': obs['tot_mask'] if resolution == "default" else obs['conv_mask'],
            'mod_flux': optimizer.best_model.detach(),
            'mod_errs': optimizer.best_model_errs.detach(),
            'loss': optimizer.loss,
        }
        np.savez(
            fits_dir.joinpath(f"{obs_name}_fit_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_{n + 1}.npz"),
            **optim_fit
        )

        ##############################
        ######## PLOT FITTING ########
        ##############################
        # Plot Convergence
        if configs['output']['plot_conv']:
            print('Plotting Convergence')
            n_panels = payne.n_cont_coeffs + 2 + np.sum(
                [optim_fit[key] is not None for key in ['rv', 'log_vmacro', 'log_vsini', 'inst_res']])
            panel = 0
            fig = plt.figure(figsize=(10, n_panels * 2))
            gs = GridSpec(n_panels, 1)
            gs.update(hspace=0.0)
            ax0 = plt.subplot(gs[panel, 0])
            ax0.plot(optimizer.history['loss'], label='loss')
            ax0.set_ylim(np.min(optimizer.history['loss']), 1.1*optimizer.history['loss'][0])
            ax0.legend()
            ax0.set_yscale('log')
            panel += 1
            ax1 = plt.subplot(gs[panel, 0], sharex=ax0)
            for i in range(optimizer.n_stellar_labels):
                ax1.plot(
                    torch.cat(optimizer.history['stellar_labels'])[:, i],
                    label=optimizer.emulator.models[0].labels[i],
                    alpha=0.5
                )
            ax1.set_ylim(-0.6, 0.6)
            ax1.legend(ncol=payne.n_stellar_labels, loc='lower center', fontsize=6)
            panel += 1
            ax2 = plt.subplot(gs[panel, 0], sharex=ax0)
            ax2.plot(torch.cat(optimizer.history['rv']).detach().numpy(), label='rv')
            ax2.legend()
            panel += 1
            if optimizer.log_vmacro is not None:
                ax = plt.subplot(gs[panel, 0])
                ax.plot(torch.cat(optimizer.history['log_vmacro']).detach().numpy(), label='log_vmacro')
                ax.legend()
                panel += 1
            if optimizer.log_vsini is not None:
                ax = plt.subplot(gs[panel, 0])
                ax.plot(torch.cat(optimizer.history['log_vsini']).detach().numpy(), label='log_vsini')
                ax.legend()
                panel += 1
            if optimizer.inst_res is not None:
                ax = plt.subplot(gs[panel, 0])
                ax.plot(torch.cat(optimizer.history['inst_res']).detach().numpy(), label='inst_res')
                ax.legend()
                panel += 1
            cont_coeffs = torch.stack(optimizer.history['cont_coeffs'])
            for i in range(optimizer.n_cont_coeffs):
                ax = plt.subplot(gs[panel, 0])
                for j in range(optimizer.n_obs_ord):
                    ax.plot(cont_coeffs[:, i, j], alpha=0.5)
                panel += 1
            plt.savefig(fig_dir.joinpath(f"{obs_name}_convergence_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_{n + 1}.png"))
            plt.close('all')

        # Plot Fits
        if configs['output']['plot_fit']:
            print('Plotting Fits')
            if resolution == "default":
                chi2 = ((optimizer.best_model[0].detach().numpy() - obs['spec']) / (
                    np.sqrt(optimizer.best_model_errs[0].detach().numpy() ** 2 + obs['errs'] ** 2))) ** 2
            else:
                chi2 = ((optimizer.best_model[0].detach().numpy() - obs['conv_spec']) / (
                    np.sqrt(optimizer.best_model_errs[0].detach().numpy() ** 2 + obs['conv_errs'] ** 2))) ** 2
            for i in range(optimizer.n_obs_ord):
                fig = plt.figure(figsize=(50, 12))
                gs = GridSpec(2, 1)
                gs.update(hspace=0.0)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[1, 0], sharex=ax1)

                plt.title(
                    f"{obs_name}, Detector: {int(obs['dets'][i])}, Order: {int(obs['ords'][i])}, Resolution: {resolution}",
                    fontsize=48)
                if resolution == "default":
                    ax1.scatter(obs['wave'][i][obs['mask'][i]], obs['spec'][i][obs['mask'][i]], c='k', marker='.',
                                alpha=0.8, )
                    ax2.scatter(obs['wave'][i][obs['mask'][i]], chi2[i][obs['mask'][i]], c='k', marker='.',
                                alpha=0.8, )
                else:
                    ax1.scatter(obs['conv_wave'][i][obs['conv_mask'][i]], obs['conv_spec'][i][obs['conv_mask'][i]],
                                c='k', marker='.', alpha=0.8, )
                    ax2.scatter(obs['conv_wave'][i][obs['conv_mask'][i]], chi2[i][obs['conv_mask'][i]], c='k',
                                marker='.', alpha=0.8, )
                ax1.plot(optimizer.obs_wave[i].detach().numpy(), optimizer.best_model[0, i].detach().numpy(), c='r', alpha=0.8)
                ax1.set_ylabel('Flux [Counts]', fontsize=36)
                ax2.set_ylabel('Chi2', fontsize=36)
                ax2.set_xlabel('Wavelength [A]', fontsize=36)
                ax1.tick_params('x', labelsize=0)
                ax1.tick_params('y', labelsize=36)
                ax2.tick_params('x', labelsize=36)
                ax2.tick_params('y', labelsize=36)
                plt.savefig(fig_dir.joinpath(f"{obs_name}_spec_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}_{int(obs['ords'][i])}_{n + 1}.png"))
                plt.close('all')

            print(f"Completed Fit {n + 1}/{n_fits} for {obs_name}_{resolution}_{'bin' if bin_errors else 'int'}{snr_tag}")
            del optimizer
            del optim_fit
            plt.close('all')
            gc.collect()