from copy import deepcopy
import torch
from torch.nested import nested_tensor
from ..utils import ensure_tensor


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
