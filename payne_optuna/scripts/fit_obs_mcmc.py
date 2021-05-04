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
    parser.add_argument("obs_name", help="Name of observation to fit")
    parser.add_argument("-o", "--orders", help="Orders to fit. List or 'all'.")
    parser.add_argument(
        "-p", "--plot", action="store_true", default=False, help="Plot QA"
    )
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

    data_dir = Path(args.data_dir)
    flats_dir = data_dir.joinpath("flats")
    obs_dir = data_dir.joinpath("obs")
    tellurics_dir = data_dir.joinpath("tellurics")
    config_dir = Path(args.config_dir)
    fig_dir = data_dir.joinpath("figures")
    fits_dir = data_dir.joinpath("fits")

    flat_files = sorted(list(flats_dir.glob("*")))
    obs_files = sorted(list(obs_dir.glob("*")))
    tellurics_file = list(tellurics_dir.glob("*"))[0]
    config_files = sorted(list(config_dir.glob("*")))

    # Load Flats
    flats = {}
    for flat_file in flat_files:
        det = int(flat_file.with_suffix("").name[-2:])
        flats[det] = MasterFlats(flat_file)

    # Load Observations
    obs_file_match = [
        obs_file for obs_file in obs_files if args.obs_name in obs_file.name
    ]
    if len(obs_file_match) > 1:
        raise RuntimeError(
            f"More than one observation matches 'obs_name' ({args.obs_name})"
        )
    elif len(obs_file_match) < 1:
        raise RuntimeError(f"No observation matches 'obs_name' ({args.obs_name})")
    else:
        obs_file = obs_file_match[0]
    print(f"Loading {obs_file}")
    obs_spec_file = obs_dir.joinpath(obs_file)
    if isinstance(args.orders, str) and args.orders.lower() == "all":
        orders_to_fit = get_all_order_numbers(obs_spec_file)
    else:
        orders_to_fit = args.orders
    obs = load_spectrum(
        spec_file=obs_spec_file,
        orders_to_fit=orders_to_fit,
        extraction="OPT",
        flats=flats,
        vel_correction="heliocentric",
    )
    # Tellurics from https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
    tellurics = pd.read_csv(
        tellurics_file,
        skiprows=3,
        header=0,
        sep="\s+",
        engine="python",
        names=["wave_min", "wave_max", "instensity", "wave_center"],
    )
    tellurics["wave_min"] *= obs["vel_corr_factor"]
    tellurics["wave_max"] *= obs["vel_corr_factor"]
    # Mannual Masks
    obs["mask"][0][obs["wave"][0] < 3860] = False  # Mask blue end of spectrum
    obs["mask"][:, :64] = False  # Mask detector edges
    obs["mask"][:, -128:] = False  # Mask detector edges
    obs["mask"][obs["spec"] < 0] = False  # Mask negative fluxes
    obs["mask"][obs["spec"] > 25e3] = False  # Mask hot pixels
    obs["mask"][
        (obs["ords"] == 66)[:, np.newaxis] & (obs["wave"] < 5365)
    ] = False  # Mask weird detector response
    obs["mask"][
        (obs["ords"] == 52)[:, np.newaxis] & (obs["wave"] > 6900)
    ] = False  # Mask weird detector response
    obs["mask"][
        (obs["ords"] == 51)[:, np.newaxis] & (obs["wave"] < 6960)
    ] = False  # Mask weird detector response
    # Mask Telluric Lines
    for line in tellurics.index:
        telluric_mask = (obs["wave"] > tellurics.loc[line, "wave_min"]) & (
            obs["wave"] < tellurics.loc[line, "wave_max"]
        )
        obs["mask"][telluric_mask] = False
    # Implement Mask
    obs["raw_errs"] = deepcopy(obs["errs"])
    obs["errs"][~obs["mask"]] = np.inf
    # Scale Blaze Function
    obs["scaled_blaz"] = (
        obs["blaz"]
        / np.quantile(obs["blaz"], 0.95, axis=1)[:, np.newaxis]
        * np.quantile(obs["spec"], 0.95, axis=1)[:, np.newaxis]
    )
    # Scale Spec by Blaze
    obs["norm_spec"] = obs["spec"] / obs["scaled_blaz"]
    obs["norm_errs"] = obs["errs"] / obs["scaled_blaz"]

    # Plot Observed Spectrum & Blaze Function
    if args.plot:
        n_ord = obs["ords"].shape[0]
        fig = plt.figure(figsize=(10, n_ord))
        gs = GridSpec(n_ord, 1)
        gs.update(hspace=0.5)
        for j, order in enumerate(obs["ords"]):
            ax = plt.subplot(gs[j, 0])
            # print(name, order)
            tellurics_in_order = tellurics[
                (
                    (tellurics["wave_min"] > np.min(obs["wave"][j]))
                    & (tellurics["wave_min"] < np.max(obs["wave"][j]))
                )
                | (
                    (tellurics["wave_max"] > np.min(obs["wave"][j]))
                    & (tellurics["wave_max"] < np.max(obs["wave"][j]))
                )
            ]
            ax.plot(
                obs["wave"][j],
                obs["scaled_blaz"][j],
                alpha=0.8,
                c="r",
                label="Scaled Blaze",
            )
            ax.scatter(
                obs["wave"][j][obs["mask"][j]],
                obs["spec"][j][obs["mask"][j]],
                alpha=0.8,
                marker=".",
                s=1,
                c="k",
                label="Observed Spectrum",
            )
            if j == 0:
                ax.set_title(args.obs_name)
                ax.legend(fontsize=8)
            for line in tellurics_in_order.index:
                ax.axvspan(
                    tellurics_in_order.loc[line, "wave_min"],
                    tellurics_in_order.loc[line, "wave_max"],
                    color="grey",
                    alpha=0.5,
                )
            ax.set_ylim(0, 3 * np.mean(obs["spec"][j][obs["mask"][j]]))
            ax.text(
                0.98,
                0.70,
                f"Order: {int(order)}",
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        plt.savefig(fig_dir.joinpath(f"{args.obs_name}_obs.png"))

    # Load Models
    models = []
    for config_file in config_files:
        models.append(load_model(config_file))

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

    # Load Optimizer Best Fit
    try:
        optim_fit = np.load(fits_dir.joinpath(f"{args.obs_name}_fit.npz"))
    except (FileNotFoundError):
        raise RuntimeError(
            f"Could not load optimizer solutions ({args.obs_name}_fit.npz)."
        )

    # Convert Obs Spectrum to Tensor
    obs['norm_spec'] = ensure_tensor(obs['norm_spec'])
    obs['norm_errs'] = ensure_tensor(obs['norm_errs'])
    obs['mask'] = ensure_tensor(obs['mask'], precision=bool)

    def gaussian_log_likelihood(pred, target, pred_errs, target_errs, mask):
        tot_vars = pred_errs ** 2 + target_errs ** 2
        loglike = -0.5 * (
                torch.log(2 * np.pi * tot_vars) + (target - pred) ** 2 / (2 * tot_vars)
        )
        return torch.sum(loglike[..., mask], axis=-1)  # .detach().numpy()

    def gaussian_log_prior(x, mu, sigma):
        return np.sum(
            np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (x - mu) ** 2 / sigma ** 2
        )

    def uniform_log_prior(theta, theta_min, theta_max):
        log_prior = np.zeros(theta.shape[0])
        log_prior[np.any((theta < theta_min) | (theta > theta_max), axis=-1)] = -np.inf
        return log_prior

    # Define Log Probability
    def log_probability(theta, model, obs):
        stellar_labels = theta
        mod_spec, mod_errs = model(
            stellar_labels=ensure_tensor(stellar_labels),
            rv=ensure_tensor(optim_fit["rv"]),
            vmacro=ensure_tensor(optim_fit["vmacro"]),
            cont_coeffs=ensure_tensor(optim_fit["cont_coeffs"]),
        )
        log_like = (
            gaussian_log_likelihood(
                pred=mod_spec,
                target=obs["norm_spec"],
                pred_errs=mod_errs,
                target_errs=obs["norm_errs"],
                mask=obs["mask"],
            )
            .detach()
            .numpy()
        )
        log_priors = uniform_log_prior(stellar_labels, -0.55, 0.55)
        return log_like + log_priors

    # Initialize Walkers
    p0 = optim_fit["stellar_labels"] + 1e-1 * np.random.randn(
        64, payne.n_stellar_labels
    )
    nwalkers, ndim = p0.shape

    # Initialize Backend
    sample_file = data_dir.joinpath(f"{args.obs_name}_mcmc_samples.h5")
    backend = emcee.backends.HDFBackend(sample_file, name=f"{args.obs_name}")
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

    #
    max_steps = int(1e5)
    index = 0
    autocorr = np.empty(max_steps)
    old_tau = np.inf
    for _ in sampler.sample(p0, iterations=max_steps, progress=True):
        if sampler.iteration % 500:
            continue
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        print(
            f"Step {sampler.iteration}: Tau = {autocorr[index]:.0f}, " + \
            f"t/100Tau = {sampler.iteration/(100*autocorr[index]):.2f}, " + \
            f"|dTau/Tau| = {np.mean(np.abs(old_tau - tau) / tau)}"
        )
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        old_tau = tau
        if converged:
            break

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(
        discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2), flat=True
    )
    unscaled_flat_samples = payne.unscale_stellar_labels(flat_samples)
    scaled_mean = flat_samples.mean(axis=0)
    scaled_std = flat_samples.std(axis=0)
    unscaled_mean = unscaled_flat_samples.mean(axis=0)
    unscaled_std = unscaled_flat_samples.std(axis=0)

    print("Sampling Summary:")
    for i, label in enumerate(payne.labels):
        if label not in ["Teff", "logg", "v_micro", "Fe"]:
            print(
                f'[{label}/Fe]\t = {unscaled_mean[i] - unscaled_mean[payne.labels.index("Fe")]:.4f} ' + \
                f'+/- {np.sqrt(unscaled_std[i]**2 + unscaled_std[payne.labels.index("Fe")]**2):.4f} ' + \
                f'({scaled_mean[i]:.4f} +/- {unscaled_std[i]:.4f})'
            )
        elif label == "Fe":
            print(
                f'[{label}/H]\t = {unscaled_mean[i]:.4f} ' + \
                f'+/- {unscaled_std[i]:.4f} ({scaled_mean[i]:.4f} +/- {scaled_std[i]:.4f})'
            )
        else:
            print(
                f'{label}\t = {unscaled_mean[i]:.4f} ' + \
                f'+/- {unscaled_std[i]:.4f} ({scaled_mean[i]:.4f} +/- {scaled_std[i]:.4f})'
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
        plt.savefig(fig_dir.joinpath(f"{args.obs_name}_mcmc_chains.png"))

        fig = corner(unscaled_flat_samples, labels=payne.labels)
        fig.savefig(fig_dir.joinpath(f"{args.obs_name}_mcmc_corner.png"))
