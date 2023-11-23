from copy import deepcopy
import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import percentile_filter
import torch
from torch.nested import nested_tensor
from ..utils import ensure_tensor
import matplotlib.pyplot as plt


def pad_obs(obs):
    obs_ = deepcopy(obs)
    n_obs_ord = len(obs['wave'])
    n_obs_pix = ensure_tensor([wave.shape[0] for wave in obs['wave']], precision=int)
    n_obs_pad = torch.max(n_obs_pix) - n_obs_pix
    print(f"Observed orders do not have the same lengths ({', '.join([str(n_pix.item()) for n_pix in n_obs_pix])}); "
          f"padding all to n_pix = {torch.max(n_obs_pix)}")
    # Wavelength
    obs_wave_ragged = nested_tensor(
        [ensure_tensor(wave, precision=torch.float64) for wave in obs['wave']]
    )
    obs_wave = obs_wave_ragged.to_padded_tensor(padding=0)
    for i in range(n_obs_ord):
        if n_obs_pad[i] > 0:
            dx = torch.mean(torch.diff(obs_wave_ragged[i]))
            obs_wave[i, -n_obs_pad[i]:] = torch.linspace(
                obs_wave[i, -n_obs_pad[i]-1],
                (obs_wave[i, -n_obs_pad[i]-1]) + (n_obs_pad[i]-1) * dx,
                n_obs_pad[i]
            )
    #obs_['wave'] = [wave.detach().numpy() for wave in obs_wave]
    obs_['wave'] = obs_wave.detach().numpy()
    # Flux
    obs_flux_ragged = nested_tensor(
        [ensure_tensor(flux, precision=torch.float64) for flux in obs['flux']]
    )
    obs_flux = obs_flux_ragged.to_padded_tensor(padding=0)
    #obs_['flux'] = [flux.detach().numpy() for flux in obs_flux]
    obs_['flux'] = obs_flux.detach().numpy()
    # Ivar
    obs_ivar_ragged = nested_tensor(
        [ensure_tensor(ivar) for ivar in obs['ivar']]
    )
    obs_ivar = obs_ivar_ragged.to_padded_tensor(padding=0)
    #obs_['ivar'] = [ivar.detach().numpy() for ivar in obs_ivar]
    obs_['ivar'] = obs_ivar.detach().numpy()
    # Mask
    obs_mask_ragged = nested_tensor(
        [ensure_tensor(mask, precision=bool) for mask in obs['mask']]
    )
    obs_mask = obs_mask_ragged.to_padded_tensor(padding=False)
    #obs_['mask'] = [mask.detach().numpy() for mask in obs_mask]
    obs_['mask'] = obs_mask.detach().numpy()
    # Errs
    obs_errs_ragged = nested_tensor(
        [ensure_tensor(errs) for errs in obs['errs']]
    )
    obs_errs = obs_errs_ragged.to_padded_tensor(padding=torch.inf)
    #obs_['errs'] = [errs.detach().numpy() for errs in obs_errs]
    obs_['errs'] = obs_errs.detach().numpy()
    # Blaz
    if obs['blaz'] is not None:
        obs_blaz_ragged = nested_tensor(
            [ensure_tensor(blaz) for blaz in obs['blaz']]
        )
        obs_blaz = obs_blaz_ragged.to_padded_tensor(padding=0)
        #obs_['blaz'] = [blaz.detach().numpy() for blaz in obs_blaz]
        obs_['blaz'] = obs_blaz.detach().numpy()
    else:
        obs_['blaz'] = None
    return obs_


def gen_blaz(model, inst_res, obs, rv, prefit_cont_window=55, plot=True):
    obs_wave = torch.vstack([ensure_tensor(wave, precision=torch.float64) for wave in obs['wave']])
    obs_flux = torch.vstack([ensure_tensor(flux) for flux in obs['flux']])
    obs_errs = torch.vstack([ensure_tensor(errs) for errs in obs['errs']])
    obs_mask = torch.vstack([ensure_tensor(mask, precision=bool) for mask in obs['mask']])
    obs_blaz = np.zeros_like(obs['wave'])

    rv = rv * torch.ones((1, 1))
    stellar_labels = torch.zeros((1, model.n_stellar_labels))
    inst_res = inst_res * torch.ones((1, 1))
    mod_flux, mod_errs = model(stellar_labels, rv, None, skip_cont=True)

    footprint = np.concatenate(
        [np.ones(prefit_cont_window), np.zeros(prefit_cont_window), np.ones(prefit_cont_window)]
    )
    tot_errs = torch.sqrt(mod_errs[0] ** 2 + obs_errs ** 2)
    scaling = obs_flux / mod_flux[0]
    scaling[~obs_mask] = 1.0
    for i in range(model.n_obs_ord):
        filtered_scaling = percentile_filter(scaling[i].detach().numpy(), percentile=25, footprint=footprint)
        filtered_scaling = percentile_filter(filtered_scaling, percentile=75, footprint=footprint)
        filtered_scaling[filtered_scaling <= 0.0] = 1.0
        p = Polynomial.fit(
            x=model.obs_norm_wave[i][obs_mask[i]].detach().numpy(),
            y=filtered_scaling[obs_mask[i]],
            deg=model.cont_deg,
            w=(tot_errs[i] ** -1)[obs_mask[i]].detach().numpy(),
            window=model.cont_wave_norm_range
        )
        obs_blaz[i] = p(model.obs_norm_wave[i]).detach().numpy()
        if plot:
            plt.figure(figsize=(20, 5))
            plt.scatter(
                obs_wave[i][obs_mask[i]].detach().numpy(),
                obs_flux[i][obs_mask[i]].detach().numpy(),
                c='k', marker='.', alpha=0.8
            )
            plt.plot(
                obs_wave[i].detach().numpy(),
                (mod_flux[0, i] * p(model.obs_norm_wave[i])).detach().numpy(),
                c='r', alpha=0.8
            )
            plt.plot(
                obs_wave[i].detach().numpy(),
                obs_blaz[i],
                c='b', alpha=0.8
            )
            plt.ylim(0, 1.5 * np.nanquantile(obs_flux[i][obs_mask[i]].detach().numpy(), 0.95))
            plt.ylabel('Flux')
            plt.xlabel('Wavelength')
            plt.show()
            plt.close('all')
    return obs_blaz
