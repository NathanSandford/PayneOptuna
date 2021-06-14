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
from payne_optuna.fitting import CompositePayneEmulator
from payne_optuna.utils import ensure_tensor

import emcee

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from corner import corner


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Fit Observed Spectrum")
    parser.add_argument("config_dir", help="Directory containing model config files")
    parser.add_argument("data_dir", help="Directory containing data files")
    parser.add_argument("star", help="Name of star to be fit")
    parser.add_argument("frame", help="specific frame to be fit")
    parser.add_argument("date", help="date of observation")
    parser.add_argument("-o", "--orders", help="Orders to fit. List or 'all'.")
    parser.add_argument("-R", "--resolution", default='default', help="Resolution to convolve and fit to.")
    parser.add_argument("-Vmacro", "--fit_vmacro", action='store_true', default=False, help="Fit vmacro.")
    parser.add_argument("-Vmicro", "--fit_vmicro", action='store_true', default=False, help="Fit vmicro (Not implemented yet).")
    parser.add_argument("-Vsini", "--fit_vsini", action='store_true', default=False, help="Fit vsini.")
    parser.add_argument("-InstRes", "--fit_inst_res", action='store_true', default=False, help="Fit inst_res.")
    parser.add_argument("-N", "--n_fits", default=1, help="Number of fits to perform w/ different starts.")
    parser.add_argument('-nlte_errs', '--use_nlte_errs', action='store_true', default=False, help="Include NLTE errors.")
    parser.add_argument('-mask', '--mask_lines', action='store_true', default=False, help="Mask bad lines.")
    parser.add_argument('-p', "--plot", action='store_true', default=False, help="Plot QA")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


class MasterFlats:
    def __init__(self, file):
        with fits.open(file) as hdul:
            self.pixelflat_model = hdul["PIXELFLAT_MODEL"].data
        self.spec_dim, self.spat_dim = self.pixelflat_model.shape
        self.spec_arr = np.arange(self.spec_dim)
        self.spat_arr = np.arange(self.spat_dim)
        self.pixelflat_model[~np.isfinite(self.pixelflat_model)] = 1e10
        self.f_flat_2d = RectBivariateSpline(
            x=self.spec_arr, y=self.spat_arr, z=self.pixelflat_model, s=0
        )

    def get_raw_blaze(self, trace_spat):
        return self.f_flat_2d.ev(self.spec_arr, trace_spat)

    def get_blaze(self, trace_spat, std=2500):
        raw_blaze = self.get_raw_blaze(trace_spat)
        if np.any((raw_blaze < 0) | (raw_blaze > 65000)):
            bad_pix = np.argwhere((raw_blaze < 0) | (raw_blaze > 65000)).flatten()[
                [0, -1]
            ] + [-5, 5]
            idx = np.r_[0 : bad_pix[0], bad_pix[1] : len(self.spec_arr)]
            # idx = np.argwhere((raw_blaze < 0) | (raw_blaze > 65000)).flatten()[-1] + 5
        else:
            idx = np.r_[0 : len(self.spec_arr)]
            # idx = 0
        f_flat_1d = UnivariateSpline(
            x=self.spec_arr[idx],
            y=raw_blaze[idx],
            s=self.spec_dim * std,
            ext="extrapolate",
        )
        return f_flat_1d(self.spec_arr)


def get_all_order_numbers(spec_file):
    with fits.open(spec_file) as hdul:
        hdul_ext = list(hdul[0].header["EXT[0-9]*"].values())
    orders = [int(ext[-4:]) for ext in hdul_ext]
    return np.array(orders)


def get_geomotion_correction(
    radec, time, longitude, latitude, elevation, refframe="heliocentric"
):
    """
    Lifted from PypeIt
    """
    loc = (
        longitude * u.deg,
        latitude * u.deg,
        elevation * u.m,
    )
    obstime = Time(time.value, format=time.format, scale="utc", location=loc)
    # Calculate ICRS position and velocity of Earth's geocenter
    ep, ev = solar_system.get_body_barycentric_posvel("earth", obstime)
    # Calculate GCRS position and velocity of observatory
    op, ov = obstime.location.get_gcrs_posvel(obstime)
    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov
    if refframe == "heliocentric":
        # ICRS position and velocity of the Sun
        sp, sv = solar_system.get_body_barycentric_posvel("sun", obstime)
        velocity += sv
    # Get unit ICRS vector in direction of SkyCoord
    sc_cartesian = radec.icrs.represent_as(UnitSphericalRepresentation).represent_as(
        CartesianRepresentation
    )
    vel = sc_cartesian.dot(velocity).to(u.km / u.s).value
    vel_corr = np.sqrt((1.0 + vel / 299792.458) / (1.0 - vel / 299792.458))
    return vel_corr


def load_spectrum(
    spec_file, orders_to_fit, extraction="OPT", flats=None, vel_correction=None
):
    with fits.open(spec_file) as hdul:
        header = hdul[0].header
        hdul_ext = list(header["EXT[0-9]*"].values())
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
            order_ext = [ext for ext in hdul_ext if f"{order:04.0f}" in ext][0]
            obs_ords[i] = int(order)
            obs_dets[i] = int(order_ext.split("DET")[1][:2])
            obs_spat[i] = hdul[order_ext].data[f"TRACE_SPAT"]
            obs_wave[i] = hdul[order_ext].data[f"{extraction.upper()}_WAVE"]
            obs_spec[i] = hdul[order_ext].data[f"{extraction.upper()}_COUNTS"]
            obs_errs[i] = hdul[order_ext].data[f"{extraction.upper()}_COUNTS_SIG"]
            obs_mask[i] = hdul[order_ext].data[f"{extraction.upper()}_MASK"]
            if flats is not None:
                obs_blaz[i] = flats[obs_dets[i]].get_blaze(obs_spat[i], std=1e5)
        if vel_correction is not None:
            vel_corr_factor = get_geomotion_correction(
                radec=SkyCoord(ra=header["RA"], dec=header["DEC"], unit=(u.deg, u.deg)),
                time=Time(header["MJD"], format="mjd"),
                longitude=header["LON-OBS"],
                latitude=header["LAT-OBS"],
                elevation=header["ALT-OBS"],
                refframe="heliocentric",
            )
            obs_wave *= vel_corr_factor
        else:
            vel_corr_factor = None
        obs_dict = {
            "ords": obs_ords,
            "dets": obs_dets,
            "spat": obs_spat,
            "wave": obs_wave,
            "spec": obs_spec,
            "errs": obs_errs,
            "mask": obs_mask,
            "blaz": obs_blaz,
            "vel_corr_factor": vel_corr_factor,
        }
        return obs_dict


def load_model(config_file):
    # Load Configs & Set Paths
    with open(config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    print(f"Loading Model {model_name}")
    input_dir = Path(configs["paths"]["input_dir"])
    output_dir = Path(configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    ckpt_file = sorted(list(ckpt_dir.glob("*.ckpt")))[-1]
    # Load Meta
    with open(meta_file) as file:
        meta = yaml.load(file, Loader=yaml.UnsafeLoader)
    # Load the Payne
    nn_model = LightningPaynePerceptron.load_from_checkpoint(
        str(ckpt_file), input_dim=meta["input_dim"], output_dim=meta["output_dim"]
    )
    nn_model.load_meta(meta)
    # Load Model Error from Validation
    try:
        validation_file = model_dir.joinpath("validation_results.npz")
        with np.load(validation_file) as tmp:
            nn_model.mod_errs = tmp["median_approx_err_wave"]
    except FileNotFoundError:
        print("validation_results.npz does not exist; assuming zero model error.")
        nn_model.mod_errs = np.zeros_like(nn_model.wavelength)
    return nn_model


def find_model_breaks(models, obs):
    model_bounds = np.zeros((len(models), 2))
    for i, mod in enumerate(models):
        model_coverage = mod.wavelength[[0, -1]]
        ord_wave_bounds_in_model = obs["wave"][:, [0, -1]][
            (obs["wave"][:, 0] > model_coverage[0])
            & (obs["wave"][:, -1] < model_coverage[-1])
        ]
        wave_min = model_coverage[0] if i == 0 else ord_wave_bounds_in_model[0, 0]
        wave_max = (
            model_coverage[-1]
            if i == len(models) - 1
            else ord_wave_bounds_in_model[-1, -1]
        )
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
    mpl.rc("axes", grid=True, lw=2)
    mpl.rc("ytick", direction="in", labelsize=10)
    mpl.rc("ytick.major", size=5, width=1)
    mpl.rc("xtick", direction="in", labelsize=10)
    mpl.rc("xtick.major", size=5, width=1)
    mpl.rc("ytick", direction="in", labelsize=10)
    mpl.rc("ytick.major", size=5, width=1)
    mpl.rc("grid", alpha=0.75, lw=1)
    mpl.rc("legend", edgecolor="k", framealpha=1, fancybox=False)
    mpl.rc("figure", dpi=300)

    # I/O Prep
    config_dir = Path(args.config_dir)
    data_dir = Path(args.data_dir)
    flats_dir = data_dir.joinpath('flats')
    obs_dir = data_dir.joinpath('obs')
    mask_dir = data_dir.joinpath('masks')
    nlte_errs_dir = data_dir.joinpath('nlte_errs')
    fig_dir = data_dir.joinpath('figures')
    fits_dir = data_dir.joinpath('fits')
    sample_dir = data_dir.joinpath('samples')

    obs_name = f'{args.star}_{args.frame}_{args.date}'

    config_files = sorted(list(config_dir.glob('*')))
    flat_files = {
        1: flats_dir.joinpath(f'MasterFlat_B_{args.date}.fits'),
        2: flats_dir.joinpath(f'MasterFlat_G_{args.date}.fits'),
        3: flats_dir.joinpath(f'MasterFlat_R_{args.date}.fits'),
    }
    obs_spec_file = obs_dir.joinpath(f'spec1d_{obs_name}.fits')
    tellurics_file = mask_dir.joinpath('tellurics.txt')
    bad_line_mask_file = mask_dir.joinpath('bad_lines.npy')
    nlte_errs_files = sorted(list(nlte_errs_dir.glob('*')))

    # Load Flats
    flats = {}
    for det, flat_file in flat_files.items():
        flats[det] = MasterFlats(flat_file)

    # Load Masks
    if args.mask_lines:
        line_masks = np.load(bad_line_mask_file)

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

    print(f'Loading {obs_spec_file.name}')
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
    print('Applying masks')
    # Mannual Masks
    obs['mask'][:, :64] = False  # Mask detector edges
    obs['mask'][:, -128:] = False  # Mask detector edges
    obs['mask'][obs['spec'] < -50] = False  # Mask negative fluxes
    obs['mask'][obs['spec'] > 25e3] = False  # Mask hot pixels
    obs['mask'][(obs['ords'] == 92)[:, np.newaxis] & (obs['wave'] < 3860)] = False  # Mask blue end of spectrum
    obs['mask'][(obs['ords'] == 66)[:, np.newaxis] & (obs['wave'] < 5365)] = False  # Mask weird detector response
    obs['mask'][(obs['ords'] == 61)[:, np.newaxis] & (obs['wave'] > 5880)] = False  # Mask weird lines
    obs['mask'][(obs['ords'] == 60)[:, np.newaxis] & (obs['wave'] < 5900)] = False  # Mask weird lines
    obs['mask'][(obs['ords'] == 52)[:, np.newaxis] & (obs['wave'] > 6900)] = False  # Mask weird detector response
    obs['mask'][(obs['ords'] == 51)[:, np.newaxis] & (obs['wave'] < 6960)] = False  # Mask weird detector response
    # Mask Telluric Lines
    # Tellurics from https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
    tellurics = pd.read_csv(tellurics_file, skiprows=3, header=0, sep='\s+', engine='python',
                            names=['wave_min', 'wave_max', 'instensity', 'wave_center'])
    tellurics['wave_min'] *= obs['vel_corr_factor']
    tellurics['wave_max'] *= obs['vel_corr_factor']
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

    # Determine Model Breaks
    model_bounds = find_model_breaks(models, obs)
    print("Model bounds determined to be:")
    [print(f"{i[0]:.2f} - {i[1]:.2f} Angstrom") for i in model_bounds]

    # Initialize Emulator
    payne = CompositePayneEmulator(
        models=models,
        model_bounds=model_bounds,
        cont_deg=6,
        cont_wave_norm_range=(-10, 10),
        obs_wave=obs["wave"],
        include_model_errs=True,
        model_res=86600,
        vmacro_method="iso",
    )

    # Convolve Observed Spectrum
    if args.resolution != "default":
        print(f'Convolving Observed Spectrum to R={args.resolution}')
        masked_spec = deepcopy(obs['spec'])
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
        # Scale Spec by Blaze
        obs["norm_spec"] = obs["conv_spec"] / obs["scaled_blaz"]
        obs["norm_errs"] = obs["conv_errs"] / obs["scaled_blaz"]
    else:
        print('Using default resolution')
        # Scale Spec by Blaze
        obs["norm_spec"] = obs["spec"] / obs["scaled_blaz"]
        obs["norm_errs"] = obs["errs"] / obs["scaled_blaz"]

    # Load Optimizer Best Fit
    fit_files = sorted(list(fits_dir.glob(f'{obs_name}_fit_{args.resolution}_*.npz')))
    if len(fit_files) == 0:
        raise RuntimeError(
            f"Could not load optimizer solutions ({obs_name}_fit.npz)."
        )
    losses = np.zeros(len(fit_files))
    for i, fit_file in enumerate(fit_files):
        tmp = np.load(fit_file)
        losses[i] = tmp['loss']
    best_idx = np.argmin(losses)
    print(f'Best from optimization: Trial {best_idx+1}')
    optim_fit = np.load(fit_files[i])

    # Convert Obs Spectrum to Tensor
    obs['norm_spec'] = ensure_tensor(obs['norm_spec'])
    obs['norm_errs'] = ensure_tensor(obs['norm_errs'])
    if args.resolution == "default":
        obs['mask'] = ensure_tensor(obs['mask'], precision=bool)
    else:
        obs['mask'] = ensure_tensor(obs['conv_mask'], precision=bool)

    # Teff & logg  Priors
    teff_mu = payne.scale_stellar_labels(4450 * torch.ones(payne.n_stellar_labels))[0]
    teff_sigma = payne.scale_stellar_labels(4500 * torch.ones(payne.n_stellar_labels))[0] \
                 - payne.scale_stellar_labels(4450 * torch.ones(payne.n_stellar_labels))[0]
    logg_mu = payne.scale_stellar_labels(0.85 * torch.ones(payne.n_stellar_labels))[1]
    logg_sigma = payne.scale_stellar_labels(0.851 * torch.ones(payne.n_stellar_labels))[1] \
                 - payne.scale_stellar_labels(0.85 * torch.ones(payne.n_stellar_labels))[1]

    def gaussian_log_likelihood(pred, target, pred_errs, target_errs, mask):
        tot_vars = pred_errs ** 2 + target_errs ** 2
        loglike = -0.5 * (
                torch.log(2 * np.pi * tot_vars) + (target - pred) ** 2 / (2 * tot_vars)
        )
        return torch.sum(loglike[..., mask], axis=-1)

    def gaussian_log_prior(x, mu, sigma):
        return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (x - mu) ** 2 / sigma ** 2


    def uniform_log_prior(theta, theta_min, theta_max):
        log_prior = np.zeros(theta.shape[0])
        log_prior[np.any((theta < theta_min) | (theta > theta_max), axis=-1)] = -np.inf
        return log_prior

    # Define Log Probability
    def log_probability(theta, model, obs):
        stellar_labels = theta[:, :model.n_stellar_labels]
        cont_coeffs = optim_fit["cont_coeffs"]
        reverse_idx = -1
        rv = ensure_tensor(theta[:, reverse_idx])
        if args.fit_vmacro:
            reverse_idx -= 1
            log_vmacro = theta[:, reverse_idx]
        else:
            log_vmacro = None
        if args.fit_vsini:
            reverse_idx -= 1
            log_vsini = theta[:, reverse_idx]
        else:
            log_vsini = None
        if args.fit_inst_res:
            reverse_idx -= 1
            inst_res = theta[:, reverse_idx]
        else:
            inst_res = None
        mod_spec, mod_errs = model(
            stellar_labels=ensure_tensor(stellar_labels),
            rv=ensure_tensor(rv),
            vmacro=ensure_tensor(10**log_vmacro) if log_vmacro is not None else None,
            vsini=ensure_tensor(10**log_vsini) if log_vsini is not None else None,
            inst_res=ensure_tensor(inst_res) if inst_res is not None else None,
            cont_coeffs=ensure_tensor(cont_coeffs),
        )
        log_like = (
            gaussian_log_likelihood(
                pred=mod_spec,
                target=obs["norm_spec"],
                pred_errs=mod_errs,
                target_errs=obs["norm_errs"],
                mask=obs["mask"],
            ).detach().numpy()
        )
        log_priors = np.zeros(theta.shape[0])
        log_priors += gaussian_log_prior(stellar_labels[:, 0], teff_mu, teff_sigma)
        log_priors += gaussian_log_prior(stellar_labels[:, 1], logg_mu, logg_sigma)
        log_priors += uniform_log_prior(stellar_labels[:, 2:], -0.55, 0.55)
        log_priors += gaussian_log_prior(rv, optim_fit["rv"], 0.01)
        if args.fit_vmacro:
            log_priors += uniform_log_prior(log_vmacro, -1, 1.3)
        if args.fit_vsini:
            log_priors += uniform_log_prior(log_vsini, -1, 1.3)
        if args.fit_inst_res:
            log_priors += gaussian_log_prior(inst_res, 1.0, 0.001)
        return log_like + log_priors

    ### Run Burn-In ###
    # Initialize Walkers
    p0_list = [optim_fit["stellar_labels"][0]]
    label_names = deepcopy(payne.labels)
    if args.fit_inst_res:
        p0_list.append(optim_fit["inst_res"][0])
        label_names.append("inst_res")
    if args.fit_vsini:
        p0_list.append(optim_fit["log_vsini"][0])
        label_names.append("log_vsini")
    if args.fit_vmacro:
        p0_list.append(optim_fit["log_vmacro"][0])
        label_names.append("log_vmacro")
    p0_list.append(optim_fit["rv"])
    label_names.append("rv")
    p0 = np.concatenate(p0_list) + 1e-2 * np.random.randn(64, len(label_names))
    nwalkers, ndim = p0.shape
    # Initialize Backend
    sample_file = sample_dir.joinpath(f"{obs_name}_{args.resolution}.h5")
    backend = emcee.backends.HDFBackend(sample_file, name=f"burn_in")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(payne, obs),
        vectorize=True,
        backend=backend,
    )
    # Run Sampler until walkers stop wandering
    p_mean_last = p0.mean(0)
    for _ in sampler.sample(p0, iterations=5000, progress=True, store=True):
        if (sampler.iteration % 100):
            continue
        p_mean = sampler.get_chain(flat=True, thin=1, discard=sampler.iteration - 100).mean(0)
        print(f'max(dMean) = {np.max(p_mean - p_mean_last)}')
        if np.abs(np.max(p_mean - p_mean_last)) < 0.001:
            p_mean_last = p_mean
            break
        p_mean_last = p_mean
    if args.plot:
        samples = sampler.get_chain()
        fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(label_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(fig_dir.joinpath(f"{obs_name}_burnin_{args.resolution}.png"))
    print('Burn-In Complete')

    ### Run For Real ###
    # Initialize Walkers
    p0 = p_mean_last + 1e-3 * np.random.randn(512, len(label_names))
    nwalkers, ndim = p0.shape
    # Initialize Backend
    backend = emcee.backends.HDFBackend(sample_file, name=f"{obs_name}")
    backend.reset(nwalkers, ndim)
    # Initialize Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(payne, obs),
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
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        print(
            f"{args.obs_name} Step {sampler.iteration}: Tau = {np.max(tau):.0f}, " +
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

    samples = sampler.get_chain()
    scaled_flat_samples = sampler.get_chain(
        discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2), flat=True
    )
    unscaled_flat_samples = payne.unscale_stellar_labels(scaled_flat_samples[:, :payne.n_stellar_labels])
    flat_samples = np.concatenate([
        unscaled_flat_samples,
        scaled_flat_samples[:, payne.n_stellar_labels:]
    ], axis=1)
    scaled_mean = scaled_flat_samples.mean(axis=0)
    scaled_std = scaled_flat_samples.std(axis=0)
    unscaled_mean = unscaled_flat_samples.mean(axis=0)
    unscaled_std = unscaled_flat_samples.std(axis=0)

    print(f"{args.obs_name} Sampling Summary:")
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
                f'{label}\t = {unscaled_mean[i]*int(args.resolution):.0f} ' +
                f'+/- {unscaled_std[i]*int(args.resolution):.0f}'
            )
        elif label in ["rv", "log_vmacro", "log_vsini"]:
            print(
                f'{label}\t = {unscaled_mean[i]:.2f} +/- {unscaled_std[i]:.2f}'
            )
        else:
            print(
                f'[{label}/Fe]\t = {unscaled_mean[i] - unscaled_mean[payne.labels.index("Fe")]:.4f} ' +
                f'+/- {np.sqrt(unscaled_std[i]**2 + unscaled_std[payne.labels.index("Fe")]**2):.4f} ' +
                f'({scaled_mean[i]:.4f} +/- {unscaled_std[i]:.4f})'
            )

    if args.plot:
        fig, axes = plt.subplots(payne.n_stellar_labels, figsize=(10, 15), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(payne.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(fig_dir.joinpath(f"{obs_name}_chains_{args.resolution}.png"))

        fig = corner(unscaled_flat_samples, labels=payne.labels)
        fig.savefig(fig_dir.joinpath(f"{obs_name}_corner_{args.resolution}.png"))
