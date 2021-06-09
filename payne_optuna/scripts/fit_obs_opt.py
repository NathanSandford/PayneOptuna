import argparse
import yaml
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, solar_system
from astropy.coordinates import UnitSphericalRepresentation, CartesianRepresentation

import torch
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.fitting import CompositePayneEmulator, PayneOptimizer, UniformLogPrior, GaussianLogPrior, FlatLogPrior
from payne_optuna.utils import ensure_tensor

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Fit Observed Spectrum")
    parser.add_argument("config_dir", help="Directory containing model config files")
    parser.add_argument("data_dir", help="Directory containing data files")
    parser.add_argument("-o", "--orders", help="Orders to fit. List or 'all'.")
    parser.add_argument("-R", "--resolution", default='default', help="Resolution to convolve and fit to.")
    parser.add_argument("-Vmacro", "--fit_vmacro", action='store_true', default=False, help="Fit vmacro.")
    parser.add_argument("-Vmicro", "--fit_vmicro", action='store_true', default=False, help="Fit vmicro (Not implemented yet).")
    parser.add_argument("-Vsini", "--fit_vsini", action='store_true', default=False, help="Fit vsini.")
    parser.add_argument("-InstRes", "--fit_inst_res", action='store_true', default=False, help="Fit inst_res.")
    parser.add_argument("-N", "--n_fits", default=1, help="Number of fits to perform w/ different starts.")
    parser.add_argument('-nlte_errs', '--use_nlte_errs', action='store_true', default=False, help="Include NLTE errors.")
    parser.add_argument('-mask', '--mask_lines', action='store_true', default=False, help="Mask individual Lines.")
    parser.add_argument('-p', "--plot", action='store_true', default=False, help="Plot QA")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


class MasterFlats:
    def __init__(self, file):
        with fits.open(file) as hdul:
            self.pixelflat_model = hdul['PIXELFLAT_MODEL'].data
        self.spec_dim, self.spat_dim = self.pixelflat_model.shape
        self.spec_arr = np.arange(self.spec_dim)
        self.spat_arr = np.arange(self.spat_dim)
        self.pixelflat_model[~np.isfinite(self.pixelflat_model)] = 1e10
        self.f_flat_2d = RectBivariateSpline(x=self.spec_arr, y=self.spat_arr, z=self.pixelflat_model, s=0)

    def get_raw_blaze(self, trace_spat):
        return self.f_flat_2d.ev(self.spec_arr, trace_spat)

    def get_blaze(self, trace_spat, std=2500):
        raw_blaze = self.get_raw_blaze(trace_spat)
        if np.any((raw_blaze < 0) | (raw_blaze > 65000)):
            bad_pix = np.argwhere((raw_blaze < 0) | (raw_blaze > 65000)).flatten()[[0, -1]] + [-5, 5]
            idx = np.r_[0:bad_pix[0], bad_pix[1]:len(self.spec_arr)]
            # idx = np.argwhere((raw_blaze < 0) | (raw_blaze > 65000)).flatten()[-1] + 5
        else:
            idx = np.r_[0:len(self.spec_arr)]
            # idx = 0
        f_flat_1d = UnivariateSpline(x=self.spec_arr[idx], y=raw_blaze[idx], s=self.spec_dim * std, ext='extrapolate')
        return f_flat_1d(self.spec_arr)


def get_all_order_numbers(spec_file):
    with fits.open(spec_file) as hdul:
        hdul_ext = list(hdul[0].header['EXT[0-9]*'].values())
    orders = [int(ext[-4:]) for ext in hdul_ext]
    return np.array(orders)


def get_geomotion_correction(radec, time, longitude, latitude, elevation, refframe='heliocentric'):
    '''
    Lifted from PypeIt
    '''
    loc = (longitude * u.deg, latitude * u.deg, elevation * u.m,)
    obstime = Time(time.value, format=time.format, scale='utc', location=loc)
    # Calculate ICRS position and velocity of Earth's geocenter
    ep, ev = solar_system.get_body_barycentric_posvel('earth', obstime)
    # Calculate GCRS position and velocity of observatory
    op, ov = obstime.location.get_gcrs_posvel(obstime)
    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov
    if refframe == "heliocentric":
        # ICRS position and velocity of the Sun
        sp, sv = solar_system.get_body_barycentric_posvel('sun', obstime)
        velocity += sv
    # Get unit ICRS vector in direction of SkyCoord
    sc_cartesian = radec.icrs.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation)
    vel = sc_cartesian.dot(velocity).to(u.km / u.s).value
    vel_corr = np.sqrt((1. + vel / 299792.458) / (1. - vel / 299792.458))
    return vel_corr


def load_spectrum(spec_file, orders_to_fit, extraction='OPT', flats=None, vel_correction=None):
    with fits.open(spec_file) as hdul:
        header = hdul[0].header
        hdul_ext = list(header['EXT[0-9]*'].values())
        n_pix = hdul[1].data.shape[0]
        obs_ords = np.zeros(len(orders_to_fit))
        obs_dets = np.zeros(len(orders_to_fit))
        obs_spat = np.zeros((len(orders_to_fit), n_pix))
        obs_wave = np.zeros((len(orders_to_fit), n_pix))
        obs_spec = np.zeros((len(orders_to_fit), n_pix))
        obs_errs = np.zeros((len(orders_to_fit), n_pix))
        obs_blaz = np.ones((len(orders_to_fit), n_pix))
        obs_mask = np.ones((len(orders_to_fit), n_pix), dtype=bool)
        for i, order in enumerate(orders_to_fit):
            order_ext = [ext for ext in hdul_ext if f'{order:04.0f}' in ext][0]
            obs_ords[i] = int(order)
            obs_dets[i] = int(order_ext.split('DET')[1][:2])
            obs_spat[i] = hdul[order_ext].data[f'TRACE_SPAT']
            obs_wave[i] = hdul[order_ext].data[f'{extraction.upper()}_WAVE']
            obs_spec[i] = hdul[order_ext].data[f'{extraction.upper()}_COUNTS']
            obs_errs[i] = hdul[order_ext].data[f'{extraction.upper()}_COUNTS_SIG']
            obs_mask[i] = hdul[order_ext].data[f'{extraction.upper()}_MASK']
            if flats is not None:
                obs_blaz[i] = flats[obs_dets[i]].get_blaze(obs_spat[i], std=1e5)
        if vel_correction is not None:
            vel_corr_factor = get_geomotion_correction(
                radec=SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg, u.deg)),
                time=Time(header['MJD'], format='mjd'),
                longitude=header['LON-OBS'],
                latitude=header['LAT-OBS'],
                elevation=header['ALT-OBS'],
                refframe='heliocentric',
            )
            obs_wave *= vel_corr_factor
        else:
            vel_corr_factor = None
        obs_dict = {
            'ords': obs_ords,
            'dets': obs_dets,
            'spat': obs_spat,
            'wave': obs_wave,
            'spec': obs_spec,
            'errs': obs_errs,
            'mask': obs_mask,
            'blaz': obs_blaz,
            'vel_corr_factor': vel_corr_factor,
        }
        return obs_dict


def load_model(config_file):
    # Load Configs & Set Paths
    with open(config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    print(f'Loading Model {model_name}')
    input_dir = Path(configs["paths"]["input_dir"])
    output_dir = Path(configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    ckpt_file = sorted(list(ckpt_dir.glob('*.ckpt')))[-1]
    # Load Meta
    with open(meta_file) as file:
        meta = yaml.load(file, Loader=yaml.UnsafeLoader)
    # Load the Payne
    nn_model = LightningPaynePerceptron.load_from_checkpoint(
        str(ckpt_file),
        input_dim=meta['input_dim'],
        output_dim=meta['output_dim']
    )
    nn_model.load_meta(meta)
    # Load Model Error from Validation
    try:
        validation_file = model_dir.joinpath('validation_results.npz')
        with np.load(validation_file) as tmp:
            nn_model.mod_errs = tmp['median_approx_err_wave']
    except FileNotFoundError:
        print('validation_results.npz does not exist; assuming zero model error.')
        nn_model.mod_errs = np.zeros_like(nn_model.wavelength)
    return nn_model


def find_model_breaks(models, obs):
    model_bounds = np.zeros((len(models), 2))
    for i, mod in enumerate(models):
        model_coverage = mod.wavelength[[0, -1]]
        ord_wave_bounds_in_model = obs['wave'][:, [0, -1]][
            (obs['wave'][:, 0] > model_coverage[0])
            & (obs['wave'][:, -1] < model_coverage[-1])
            ]
        wave_min = model_coverage[0] if i == 0 else ord_wave_bounds_in_model[0, 0]
        wave_max = model_coverage[-1] if i == len(models) - 1 else ord_wave_bounds_in_model[-1, -1]
        model_bounds[i] = [wave_min, wave_max]
    breaks = (model_bounds.flatten()[2::2] + model_bounds.flatten()[1:-1:2]) / 2
    model_bounds[:-1, -1] = breaks
    model_bounds[1:, 0] = breaks
    model_bounds = [model_bounds[i] for i in range(len(models))]
    return model_bounds


def main(args):
    # Set Tensor Type
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Plotting Configs
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

    data_dir = Path(args.data_dir)
    flats_dir = data_dir.joinpath('flats')
    obs_dir = data_dir.joinpath('obs')
    tellurics_dir = data_dir.joinpath('tellurics')
    line_mask_dir = data_dir.joinpath('line_masks')
    nlte_errs_dir = data_dir.joinpath('nlte_errs')
    config_dir = Path(args.config_dir)
    fig_dir = data_dir.joinpath('figures')
    fits_dir = data_dir.joinpath('fits')

    flat_files = sorted(list(flats_dir.glob('*')))
    obs_files = sorted(list(obs_dir.glob('*')))
    tellurics_file = list(tellurics_dir.glob('*'))[0]
    config_files = sorted(list(config_dir.glob('*')))
    line_mask_file = list(line_mask_dir.glob('*'))[0]
    nlte_errs_files = sorted(list(nlte_errs_dir.glob('*')))

    # Load Flats
    flats = {}
    for flat_file in flat_files:
        det = int(flat_file.with_suffix('').name[-2:])
        flats[det] = MasterFlats(flat_file)

    # Load Masks
    if args.mask_lines:
        line_masks = np.load(line_mask_file)

    # Load Observations
    all_obs = {}
    for obs_file in obs_files:
        print(f'Loading {obs_file}')
        obs_spec_file = obs_dir.joinpath(obs_file)
        if isinstance(args.orders, str) and args.orders.lower() == 'all':
            orders_to_fit = get_all_order_numbers(obs_spec_file)
        else:
            orders_to_fit = args.orders
        obs = load_spectrum(
            spec_file=obs_spec_file,
            orders_to_fit=orders_to_fit,
            extraction='OPT',
            flats=flats,
            vel_correction='heliocentric',
        )
        # Tellurics from https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
        tellurics = pd.read_csv(tellurics_file, skiprows=3, header=0, sep='\s+', engine='python',
                                names=['wave_min', 'wave_max', 'instensity', 'wave_center'])
        tellurics['wave_min'] *= obs['vel_corr_factor']
        tellurics['wave_max'] *= obs['vel_corr_factor']
        # Mannual Masks
        obs['mask'][:, :64] = False  # Mask detector edges
        obs['mask'][:, -128:] = False  # Mask detector edges
        #obs['mask'][obs['spec'] < 0] = False  # Mask negative fluxes
        obs['mask'][obs['spec'] > 25e3] = False  # Mask hot pixels
        obs['mask'][(obs['ords'] == 92)[:, np.newaxis] & (obs['wave'] < 3860)] = False  # Mask blue end of spectrum
        obs['mask'][(obs['ords'] == 66)[:, np.newaxis] & (obs['wave'] < 5365)] = False  # Mask weird detector response
        obs['mask'][(obs['ords'] == 61)[:, np.newaxis] & (obs['wave'] > 5880)] = False  # Mask weird lines
        obs['mask'][(obs['ords'] == 60)[:, np.newaxis] & (obs['wave'] < 5900)] = False  # Mask weird lines
        obs['mask'][(obs['ords'] == 52)[:, np.newaxis] & (obs['wave'] > 6900)] = False  # Mask weird detector response
        obs['mask'][(obs['ords'] == 51)[:, np.newaxis] & (obs['wave'] < 6960)] = False  # Mask weird detector response
        # Mask Telluric Lines
        for line in tellurics.index:
            telluric_mask = (obs['wave'] > tellurics.loc[line, 'wave_min']) & (
                        obs['wave'] < tellurics.loc[line, 'wave_max'])
            obs['mask'][telluric_mask] = False
        # Implement Mask
        obs['raw_errs'] = deepcopy(obs['errs'])
        obs['errs'][~obs['mask']] = np.inf
        # Scale Blaze Function
        obs['scaled_blaz'] = obs['blaz'] / np.quantile(obs['blaz'], 0.95, axis=1)[:, np.newaxis] * np.quantile(
            obs['spec'], 0.95, axis=1)[:, np.newaxis]
        # Save to Dictionary
        all_obs[obs_spec_file.stem[7:]] = obs

    # Load Models
    models = []
    for i, config_file in enumerate(config_files):
        model = load_model(config_file)
        # Manually Mask Lines
        if args.mask_lines:
            for j in range(len(line_masks)):
                model.mod_errs[
                    (model.wavelength > line_masks[j][0] - line_masks[j][1])
                    & (model.wavelength < line_masks[j][0] + line_masks[j][1])
                    ] = 1
        # Add NLTE Errors
        if args.use_nlte_errs:
            try:
                nlte_errs = np.load(nlte_errs_files[i])
                model.mod_errs = np.sqrt(model.mod_errs ** 2 + nlte_errs['errs'] ** 2)
            except FileNotFoundError:
                print('NLTE error could not be loaded. Assuming zero NLTE errors.')
        models.append(model)

    # Perform Fit
    for name, obs in all_obs.items():
        print(f'Beginning Analysis of {name}')

        # Determine Model Breaks
        model_bounds = find_model_breaks(models, obs)
        print('Model bounds determined to be:')
        [print(f'{i[0]:.2f} - {i[1]:.2f} Angstrom') for i in model_bounds]

        # Initialize Emulator
        payne = CompositePayneEmulator(
            models=models,
            model_bounds=model_bounds,
            cont_deg=6,
            cont_wave_norm_range=(-10, 10),
            obs_wave=obs['wave'],
            include_model_errs=True,
            model_res=86600,
            vmacro_method='iso',
        )

        # Convolve Observed Spectrum
        if args.resolution != "default":
            print(f'Convolving Observed Spectrum to R={args.resolution}')
            masked_spec= deepcopy(obs['spec'])
            masked_spec[~obs['mask']] = obs['scaled_blaz'][~obs['mask']]
            conv_obs_flux, conv_obs_errs = payne.inst_broaden(
                wave=ensure_tensor(obs['wave'], precision=torch.float64),
                flux=ensure_tensor(masked_spec).unsqueeze(0),
                errs=ensure_tensor(obs['raw_errs']).unsqueeze(0),
                inst_res=ensure_tensor(int(args.resolution)),
                model_res=ensure_tensor(86600),
            )
            conv_obs_mask, _ = payne.inst_broaden(
                wave=ensure_tensor(obs['wave'], precision=torch.float64),
                flux=ensure_tensor(obs['mask']).unsqueeze(0),
                errs=None,
                inst_res=ensure_tensor(int(args.resolution)),
                model_res=ensure_tensor(86600),
            )
            conv_obs_mask = (conv_obs_mask > 0.999)
            conv_obs_errs[~conv_obs_mask] = np.inf
            obs['conv_spec'] = conv_obs_flux.squeeze().detach().numpy()
            obs['conv_errs'] = conv_obs_errs.squeeze().detach().numpy()
            obs['conv_mask'] = conv_obs_mask.squeeze().detach().numpy()

        # Plot Observed Spectrum & Blaze Function
        if args.plot:
            n_ord = obs['ords'].shape[0]
            fig = plt.figure(figsize=(10, n_ord))
            gs = GridSpec(n_ord, 1)
            gs.update(hspace=0.5)
            for j, order in enumerate(obs['ords']):
                ax = plt.subplot(gs[j, 0])
                tellurics_in_order = tellurics[
                    (
                            (tellurics['wave_min'] > np.min(obs['wave'][j]))
                            & (tellurics['wave_min'] < np.max(obs['wave'][j]))
                    )
                    | (
                            (tellurics['wave_max'] > np.min(obs['wave'][j]))
                            & (tellurics['wave_max'] < np.max(obs['wave'][j]))
                    )
                    ]
                ax.plot(obs['wave'][j], obs['scaled_blaz'][j], alpha=0.8, c='r', label='Scaled Blaze')
                ax.scatter(
                    obs['wave'][j][obs['mask'][j]], obs['spec'][j][obs['mask'][j]],
                    alpha=0.8, marker='.', s=1, c='k', label='Observed Spectrum'
                )
                if args.resolution != "default":
                    ax.scatter(
                        obs['wave'][j][obs['conv_mask'][j]], obs['conv_spec'][j][obs['conv_mask'][j]],
                        alpha=0.8, marker='.', s=1, c='b', label='Convolved Spectrum'
                    )
                if j == 0:
                    ax.set_title(name)
                    ax.legend(fontsize=8)
                for line in tellurics_in_order.index:
                    ax.axvspan(tellurics_in_order.loc[line, 'wave_min'], tellurics_in_order.loc[line, 'wave_max'],
                               color='grey', alpha=0.5)
                ax.set_ylim(0, 3 * np.mean(obs['spec'][j][obs['mask'][j]]))
                ax.text(0.98, 0.70, f"Order: {int(order)}", transform=ax.transAxes, fontsize=6, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.8))
            plt.savefig(fig_dir.joinpath(f'{name}_obs_{args.resolution}.png'))

        # Set Priors
        logg_mu = payne.scale_stellar_labels(0.50 * torch.ones(payne.n_stellar_labels))[1]
        logg_sigma = payne.scale_stellar_labels(0.501 * torch.ones(payne.n_stellar_labels))[1] \
                     - payne.scale_stellar_labels(0.50 * torch.ones(payne.n_stellar_labels))[1]
        priors = {
            'stellar_labels': [
                UniformLogPrior(lab, -0.55, 0.55)
                if lab != 'logg'
                else GaussianLogPrior(lab, logg_mu, logg_sigma)
                for lab in payne.labels
            ],
            'log_vmacro': UniformLogPrior('log_vmacro', -2, 1.3),
            'log_vsini': FlatLogPrior('log_vsini'),
            'inst_res': FlatLogPrior('inst_res') if args.resolution == 'default' else
                GaussianLogPrior('inst_res', int(args.resolution), 0.01*int(args.resolution))
        }

        for n in range(int(args.n_fits)):
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
                    d_log_vmacro=np.inf,
                    d_log_vsini=np.inf,
                    d_inst_res=1e0,
                    d_rv=1e-4,
                    d_cont=5e-2,
                    d_loss=-np.inf,
                    loss=-np.inf,
                ),
            )
            # Init Parameters
            stellar_labels0 = torch.FloatTensor(1, payne.n_stellar_labels).uniform_(-0.5, 0.5)
            rv0 = 'prefit'
            log_vmacro0 = torch.FloatTensor(1, 1).uniform_(-2.0, 1.3) if args.fit_vmacro else None
            log_vsini0 = torch.FloatTensor(1, 1).uniform_(-1.0, 1.0) if args.fit_vsini else None
            if args.fit_inst_res:
                inst_res0 = ensure_tensor(payne.model_res) if args.resolution == 'default' else ensure_tensor(int(args.resolution))
                inst_res0 = torch.min(
                    inst_res0 * torch.FloatTensor(1, 1).uniform_(0.9, 1.1),
                    ensure_tensor(payne.model_res)
                )
            else:
                inst_res0 = None if args.resolution == 'default' else ensure_tensor(int(args.resolution))
            # Begin Fit
            optimizer.fit(
                obs_flux=obs['spec'] if args.resolution == "default" else obs['conv_spec'],
                obs_errs=obs['errs'] if args.resolution == "default" else obs['conv_errs'],
                obs_wave=obs['wave'],
                obs_blaz=obs['scaled_blaz'],
                #obs_mask=obs['mask'],
                params=dict(
                    stellar_labels='fit',
                    rv='fit',
                    log_vmacro='fit' if args.fit_vmacro else 'const',
                    log_vsini='fit' if args.fit_vsini else 'const',
                    inst_res='fit' if args.fit_inst_res else 'const',
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
                max_epochs=10000,
                prefit_cont_window=55,
                verbose=True,
                plot_prefits=True,
                plot_fit_every=None,
            )
        
            # Unscale Stellar Labels
            unscaled_stellar_labels = payne.unscale_stellar_labels(optimizer.stellar_labels)

            print(f'Best Fit Labels ({n + 1}/{args.n_fits}):')
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
                print(f'log(vmacro)\t = {optimizer.log_vmacro.item():.2f} (km/s)')
            if optimizer.log_vsini is not None:
                print(f'log(vsini)\t = {optimizer.log_vsini.item():.2f} (km/s)')
            if optimizer.inst_res is not None:
                print(f'inst_res\t = {optimizer.inst_res.item():.2f} (km/s)')

            # Save Labels & Fits
            optim_fit = {
                'stellar_labels': optimizer.stellar_labels.detach(),
                'rv': optimizer.rv.detach(),
                'log_vmacro': optimizer.log_vmacro.detach() if optimizer.log_vmacro is not None else None,
                'log_vsini': optimizer.log_vsini.detach() if optimizer.log_vsini is not None else None,
                'inst_res': optimizer.inst_res.detach() if optimizer.inst_res is not None else None,
                'cont_coeffs':torch.stack(optimizer.cont_coeffs).detach(),
                'obs_wave': optimizer.obs_wave.detach(),
                'obs_flux': optimizer.obs_flux.detach(),
                'obs_errs': optimizer.obs_errs.detach(),
                'obs_blaz': optimizer.obs_blaz.detach(),
                'mod_flux': optimizer.best_model.detach(),
                'mod_errs': optimizer.best_model_errs.detach(),
                'loss': optimizer.loss,
            }
            np.savez(
                fits_dir.joinpath(f'{name}_fit_{args.resolution}_{n+1}.npz'),
                **optim_fit
            )

            # Plot Convergence
            if args.plot:
                n_panels = payne.n_cont_coeffs + 2 + np.sum(
                    [optim_fit[key] is not None for key in ['rv', 'log_vmacro', 'log_vsini', 'inst_res']])
                panel = 0
                fig = plt.figure(figsize=(10, n_panels * 2))
                gs = GridSpec(n_panels, 1)
                gs.update(hspace=0.0)
                ax0 = plt.subplot(gs[panel, 0])
                ax0.plot(optimizer.history['loss'], label='loss')
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
                ax2.plot(np.array(optimizer.history['rv']), label='rv')
                ax2.legend()
                panel += 1
                if optimizer.log_vmacro is not None:
                    ax = plt.subplot(gs[panel, 0])
                    ax.plot(np.array(optimizer.history['log_vmacro']), label='log_vmacro')
                    ax.legend()
                    panel += 1
                if optimizer.log_vsini is not None:
                    ax = plt.subplot(gs[panel, 0])
                    ax.plot(np.array(optimizer.history['log_vsini']), label='log_vsini')
                    ax.legend()
                    panel += 1
                if optimizer.inst_res is not None:
                    ax = plt.subplot(gs[panel, 0])
                    ax.plot(np.array(optimizer.history['inst_res']), label='inst_res')
                    ax.legend()
                    panel += 1
                cont_coeffs = torch.stack(optimizer.history['cont_coeffs'])
                for i in range(optimizer.n_cont_coeffs):
                    ax = plt.subplot(gs[panel, 0])
                    for j in range(optimizer.n_obs_ord):
                        ax.plot(cont_coeffs[:, i, j], alpha=0.5)
                    panel += 1
                plt.savefig(fig_dir.joinpath(f'{name}_convergence_{args.resolution}_{n+1}.png'))
        
            # Plot Fits
            if args.plot:
                if args.resolution == "default":
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
            
                    plt.title(f"{name}, Detector: {int(obs['dets'][i])}, Order: {int(obs['ords'][i])}, Resolution: {args.resolution}", fontsize=48)
                    if args.resolution == "default":
                        ax1.scatter(obs['wave'][i][obs['mask'][i]], obs['spec'][i][obs['mask'][i]], c='k', marker='.', alpha=0.8, )
                        ax2.scatter(obs['wave'][i][obs['mask'][i]], chi2[i][obs['mask'][i]], c='k', marker='.', alpha=0.8, )
                    else:
                        ax1.scatter(obs['wave'][i][obs['conv_mask'][i]], obs['conv_spec'][i][obs['conv_mask'][i]], c='k', marker='.', alpha=0.8, )
                        ax2.scatter(obs['wave'][i][obs['conv_mask'][i]], chi2[i][obs['conv_mask'][i]], c='k', marker='.', alpha=0.8, )
                    ax1.plot(obs['wave'][i], optimizer.best_model[0, i].detach().numpy(), c='r', alpha=0.8)
                    ax1.set_ylabel('Flux [Counts]', fontsize=36)
                    ax2.set_ylabel('Chi2', fontsize=36)
                    ax2.set_xlabel('Wavelength [A]', fontsize=36)
                    ax1.tick_params('x', labelsize=0)
                    ax1.tick_params('y', labelsize=36)
                    ax2.tick_params('x', labelsize=36)
                    ax2.tick_params('y', labelsize=36)
                    plt.savefig(fig_dir.joinpath(f"{name}_spec_{args.resolution}_{int(obs['ords'][i])}.png"))

            print(f'Completed Fit {n+1}/{args.n_fits} for {name}')