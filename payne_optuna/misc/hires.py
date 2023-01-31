import yaml
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.ndimage import percentile_filter
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from payne_optuna.utils import get_geomotion_correction, find_runs


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
        mask = np.ones_like(raw_blaze, dtype=bool)
        if np.any((raw_blaze < 0) | (raw_blaze > 1e5)):
            bad_pix_reg = find_runs((raw_blaze < 0) | (raw_blaze > 1e5), 1)
            bad_pix_reg_ext = np.array([reg + [max([-reg[0],-20]), 20] for reg in bad_pix_reg])
            idx = np.concatenate([np.r_[reg[0]:reg[1]] for reg in bad_pix_reg_ext])
            idx = idx[(idx >= 0) & (idx < len(raw_blaze))]
            mask[idx] = False
        f_flat_1d = UnivariateSpline(x=self.spec_arr[mask], y=raw_blaze[mask], s=self.spec_dim * std, ext='extrapolate')
        return f_flat_1d(self.spec_arr)


def get_all_order_numbers(spec_file):
    with fits.open(spec_file) as hdul:
        hdul_ext = list(hdul[0].header['EXT[0-9]*'].values())
    orders = [int(ext[-4:]) for ext in hdul_ext]
    return np.array(orders)


def load_spectrum(
        spec_file, orders_to_fit, extraction='OPT', flats=None, vel_correction=None
):
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
                try:
                    obs_blaz[i] = flats[obs_dets[i]].get_blaze(obs_spat[i], std=1e5)
                except:
                    print(f'Flat Failed for order {obs_ords[i]} --- Adopting flat blaze')
                    obs_blaz[i] = 1.0
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
        scl_blaz = obs_blaz / np.quantile(obs_blaz, 0.95, axis=1)[:, np.newaxis] * \
                   np.quantile(obs_spec, 0.95, axis=1)[:, np.newaxis]
        obs_dict = {
            'ords': obs_ords,
            'dets': obs_dets,
            'spat': obs_spat,
            'wave': obs_wave,
            'spec': obs_spec,
            'errs': obs_errs,
            'mask': obs_mask,
            'blaz': obs_blaz,
            'scaled_blaz': scl_blaz,
            'vel_corr_factor': vel_corr_factor,
        }
        return obs_dict


def get_ovrlp_mask(obs):
    ovrlp_mask = np.ones_like(obs['mask'], dtype=bool)
    for j in range(len(obs['ords']) - 1):
        blue_max = np.max(obs['wave'][j][obs['mask'][j]])
        red_min = np.min(obs['wave'][j + 1][obs['mask'][j + 1]])
        overlap_mid = np.mean([blue_max, red_min])
        if blue_max - red_min > 0:
            blu_cut = np.argwhere(obs['wave'][j] < overlap_mid)[-1][0]
            red_cut = np.argwhere(obs['wave'][j + 1] > overlap_mid)[0][0]
            ovrlp_mask[j, blu_cut:] = False
            ovrlp_mask[j + 1, :red_cut] = False
    return ovrlp_mask


def get_det_mask(obs, mask_left_pixels=64, mask_right_pixels=128, masks_by_order_wave={}):
    det_mask = np.ones_like(obs['mask'], dtype=bool)
    # Mask All Left Detector Edges
    if mask_left_pixels > 0:
        det_mask[:, 0:mask_left_pixels] = False
    # Mask All Right Detector Edges
    if mask_right_pixels > 0:
        det_mask[:, -mask_right_pixels:det_mask.shape[1]] = False
    # Mask Individual Orders by Wavelength
    for mask, order_mask in masks_by_order_wave.items():
        for order, cutoff in order_mask.items():
            for min_max, wave in cutoff.items():
                if min_max == "min":
                    det_mask[(obs['ords'] == order)[:, np.newaxis] & (obs['wave'] < wave)] = False
                elif min_max == "max":
                    det_mask[(obs['ords'] == order)[:, np.newaxis] & (obs['wave'] > wave)] = False
                elif min_max == "minmax":
                    det_mask[(obs['ords'] == order)[:, np.newaxis] & (obs['wave'] > wave[0]) & (
                                obs['wave'] < wave[1])] = False
                else:
                    raise KeyError(f"Key must be 'min', 'max', or 'min_max' --- not {min_max}")
    return det_mask


def get_pix_mask(obs, pos_excess=1.2, neg_excess=-0.2, window=256, percentile=95, exclude_orders=[]):
    pix_mask = deepcopy(obs['mask'])
    for i, order in enumerate(obs['ords']):
        if order not in exclude_orders:
            filtered_scaling = percentile_filter(
                (obs['spec'][i] / obs['scaled_blaz'][i]),
                percentile=percentile,
                size=window,
            )
            pix_mask[i][(obs['spec'][i] / obs['scaled_blaz'][i]) / filtered_scaling > pos_excess] = False
            pix_mask[i][(obs['spec'][i] / obs['scaled_blaz'][i]) / filtered_scaling < neg_excess] = False
    return pix_mask

def get_tell_mask(obs, telluric_file):
    tell_mask = np.ones_like(obs['mask'], dtype=bool)
    if telluric_file is False:
        return tell_mask
    tellurics = pd.read_csv(telluric_file, skiprows=3, header=0, sep='\s+', engine='python',
                            names=['wave_min', 'wave_max', 'instensity', 'wave_center'])
    tellurics['wave_min'] *= obs['vel_corr_factor']
    tellurics['wave_max'] *= obs['vel_corr_factor']
    for line in tellurics.index:
        telluric_mask = (obs['wave'] > tellurics.loc[line, 'wave_min']) & (
                obs['wave'] < tellurics.loc[line, 'wave_max'])
        tell_mask[telluric_mask] = False
    return tell_mask






